"""
corpus/adapters.py

Dataset adapters for the replay harness.

An adapter's only job is to turn a source dataset into the shape the harness
walks: a time-ordered stream of (timestamp, features, y_true), plus a
train/reference/replay split that never overlaps.

Real-data adapters deliberately do NOT bundle or download data. The
build process forbids committing anything not marked `fetch: repo`, and
several sources (Fannie Mae, TableShift's gated constituents) cannot be
fetched without credentials at all. Each adapter therefore declares where its
data must be placed and raises a clear error if it is absent, rather than
silently producing an empty corpus.

`SyntheticAdapter` needs no data and exists so the harness contract is
testable and reproducible in CI without acquiring a single licence.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

__all__ = [
    "REGISTRY",
    "BackblazeAdapter",
    "DatasetAdapter",
    "FhfaPudbAdapter",
    "FolktablesAdapter",
    "HmdaAdapter",
    "NasaPCoEAdapter",
    "Row",
    "Split",
    "SyntheticAdapter",
    "get_adapter",
]

DATA_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "data"


@dataclass(frozen=True)
class Row:
    """One observation. `y_true` is what makes replay labelling possible at all
    -- in production this is absent by construction, which is why the performance estimator has
    to estimate performance rather than measure it."""
    ts: datetime
    features: dict[str, float | str]
    y_true: int | float


@dataclass
class Split:
    """Three disjoint, ordered slices.

    train   -> fits the base model
    reference -> builds the reference window (the drift baseline)
    replay  -> the stream the harness walks

    Disjointness is not a nicety. A base model that saw its replay data is not
    degrading, it is remembering, and every label computed downstream would be
    wrong.
    """
    train: list[Row] = field(default_factory=list)
    reference: list[Row] = field(default_factory=list)
    replay: list[Row] = field(default_factory=list)

    def validate(self) -> None:
        if not (self.train and self.reference and self.replay):
            raise ValueError("all three slices must be non-empty")
        last_train = max(r.ts for r in self.train)
        first_ref = min(r.ts for r in self.reference)
        last_ref = max(r.ts for r in self.reference)
        first_replay = min(r.ts for r in self.replay)
        if first_ref < last_train:
            raise ValueError("reference overlaps train in time")
        if first_replay < last_ref:
            raise ValueError("replay overlaps reference in time")
        # The reference-window minimum bar, enforced not assumed
        if len(self.reference) < 2000:
            raise ValueError(
                f"reference window has {len(self.reference)} rows, "
                "the reference window requires N >= 2000")
        span_h = (last_ref - first_ref).total_seconds() / 3600
        if span_h < 24:
            raise ValueError(
                f"reference window spans {span_h:.1f}h, the reference window requires >= 24h")


class DatasetAdapter(Protocol):
    dataset_id: str
    targets: Sequence[str]
    numeric_features: Sequence[str]
    categorical_features: Sequence[str]

    def load(self, target: str) -> Split: ...


class SyntheticAdapter:
    """Generates a stream with controllable, known covariate drift.

    Not a stand-in for a real source in the corpus -- it adds no independent
    group and must never be counted in `G`. It exists so the harness contract
    can be tested end-to-end and reproducibly in CI without acquiring data.
    """

    dataset_id = "synthetic"
    targets = ("binary",)
    numeric_features = ("f_num_0", "f_num_1", "f_num_2", "f_num_3")
    categorical_features = ("f_cat_0",)

    # Sizing is driven by TWO constraints that pull in opposite directions,
    # and getting either wrong produces a corpus of empty windows:
    #
    #   * The reference bar: N >= 2000 AND >= 24h of coverage.
    #   * The 5-minute emission cadence: each window needs enough rows
    #     to compute moments, ECE and quantiles, or it falls below the minimum
    #     emission bar and no vector is emitted at all.
    #
    # A sparse stream satisfies the first and fails the second. Defaults below
    # are expressed in WINDOWS x ROWS-PER-WINDOW so both hold by construction:
    # 5s inter-arrival gives 60 rows per 5-minute window, and 25h of reference
    # coverage follows from the window count.
    def __init__(self, *, rows_per_window: int = 60,
                 train_windows: int = 240, reference_windows: int = 300,
                 replay_windows: int = 200, window_seconds: int = 300,
                 seed: int = 0):
        self.rows_per_window = rows_per_window
        self.window_seconds = window_seconds
        self.cadence_seconds = window_seconds / rows_per_window
        self.n_train = train_windows * rows_per_window
        self.n_reference = reference_windows * rows_per_window
        self.n_replay = replay_windows * rows_per_window
        self.seed = seed

    def _gen(self, rng: random.Random, n: int, start: datetime,
             shift: float = 0.0) -> list[Row]:
        rows = []
        ts = start
        for _ in range(n):
            x = [rng.gauss(shift, 1.0) for _ in range(4)]
            cat = rng.choice(["a", "b", "c", "d", "e"])
            logit = 0.9 * x[0] - 0.6 * x[1] + 0.3 * x[2] + (0.4 if cat in ("a", "b") else -0.2)
            p = 1 / (1 + math.exp(-logit))
            y = 1 if rng.random() < p else 0
            rows.append(Row(
                ts=ts,
                features={"f_num_0": x[0], "f_num_1": x[1], "f_num_2": x[2],
                          "f_num_3": x[3], "f_cat_0": cat},
                y_true=y,
            ))
            ts += timedelta(seconds=self.cadence_seconds)
        return rows

    def load(self, target: str = "binary") -> Split:
        rng = random.Random(self.seed)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        train = self._gen(rng, self.n_train, t0)
        ref_start = train[-1].ts + timedelta(seconds=self.cadence_seconds)
        reference = self._gen(rng, self.n_reference, ref_start)
        rep_start = reference[-1].ts + timedelta(seconds=self.cadence_seconds)
        replay = self._gen(rng, self.n_replay, rep_start)
        s = Split(train=train, reference=reference, replay=replay)
        s.validate()
        return s


class _RealDataAdapter:
    """Base for adapters over real sources. Fails loudly when data is absent."""

    dataset_id = "override-me"
    expected_path = "override-me"
    acquisition_note = ""

    def _require(self) -> Path:
        p = DATA_ROOT / self.expected_path
        if not p.exists():
            raise FileNotFoundError(
                f"{self.dataset_id}: expected data at {p}\n"
                f"  {self.acquisition_note}\n"
                f"  Nothing is auto-downloaded: the corpus build forbids "
                f"committing sources not marked `fetch: repo`, and several "
                f"sources cannot be fetched without credentials."
            )
        return p


class FolktablesAdapter(_RealDataAdapter):
    dataset_id = "folktables"
    expected_path = "folktables"
    acquisition_note = (
        "pip install folktables, then materialise ACS PUMS per state/year. "
        "EXCLUDE the bundled Adult reconstruction subsample -- it carries IPUMS "
        "terms (replication only, no redistribution), unlike the MIT package."
    )
    targets = ("income", "employment", "coverage")
    numeric_features = ("AGEP", "WKHP", "SCHL")
    categorical_features = ("COW", "MAR", "RAC1P", "SEX")

    def load(self, target: str) -> Split:  # pragma: no cover - needs real data
        self._require()
        raise NotImplementedError(
            "Folktables has survey-year granularity and no natural wall-clock "
            "time. Synthetic timestamps at a fixed cadence must be assigned "
            "here, and the run must record that the resulting 24h reference-window "
            "coverage is NOMINAL, not physical. Implement with that recorded.")


class BackblazeAdapter(_RealDataAdapter):
    dataset_id = "backblaze"
    expected_path = "backblaze"
    acquisition_note = (
        "Download Drive Stats quarterly CSVs. Licence: free, citation "
        "REQUIRED, derivative works sellable, the data itself is not. Do not "
        "commit raw CSVs; fetch at build time."
    )
    targets = ("failure",)
    numeric_features = ("smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw")
    categorical_features = ("model",)

    def load(self, target: str) -> Split:  # pragma: no cover - needs real data
        self._require()
        raise NotImplementedError(
            "Backblaze is the corpus's ONLY censored time-to-event source "
            "Right-censoring must be preserved: drives surviving "
            "the observation window are signal for survival modelling, "
            "not rows to drop.")


class HmdaAdapter(_RealDataAdapter):
    """HMDA Loan/Application Register -- the corpus's regime-break source.

    Acquired by a separate acquisition script (2007-2017, all-records,
    codes). Legal basis: 17 U.S.C. s105 and *Feist*, no grant required.

    Why this source carries the regime break: lending standards collapsed
    between 2007 and 2009. A model fitted on 2007 approval behaviour and
    replayed through 2008-2009 degrades because the world changed, not because
    the data was perturbed. That is the one shift class fault injection cannot
    manufacture (fault injection refuses to fabricate a crisis tape), and it is why
    `adversarial_regime_replay` raised NotImplementedError until this landed.

    Three decisions are recorded here rather than left implicit:

    TARGET. `action_taken` has eight values. Approved is {1 originated,
    2 approved-not-accepted, 8 preapproval-approved}; denied is {3 denied,
    7 preapproval-denied}. Values 4 (withdrawn) and 5 (file closed
    incomplete) are dropped because they are APPLICANT actions, not lender
    decisions -- scoring them as denials teaches applicant behaviour as
    though it were credit policy. Value 6 (loan purchased by the
    institution) is dropped because those rows are loans bought from other
    lenders and carry no application decision at all; labelling them would
    be a fabricated label, which is the error class this project has caught
    four times already.

    TIME. HMDA has `as_of_year` and nothing finer. As with other year-granular sources the harness
    assigns synthetic timestamps at a fixed cadence, and the 24 h
    reference-window bar is then satisfied BY CONSTRUCTION -- its satisfaction
    is **nominal, not physical**, exactly as recorded for Folktables. Evaluation
    must not read HMDA's temporal coverage as real elapsed time.

    GEOGRAPHY. `state_code` is the only geographic feature used. `county_code`
    and `msamd` are present in the acquired files but deliberately NOT fed to
    the model: county-level geography combined with income and loan amount is a
    re-identification vector, and the legal basis carries HMDA's prohibition on
    re-identifying borrowers forward as a use restriction independent of
    copyright. State-level captures the regional character of the crisis
    without that exposure.
    """

    dataset_id = "hmda"
    expected_path = "hmda"
    acquisition_note = (
        "Download the HMDA LAR files into <dir> as "
        "hmda_YYYY.csv.gz plus a manifest.json recording the source and terms of use. "
        "Point HMDA_DATA_DIR at that directory if it is not under corpus/data."
    )
    targets = ("approval",)

    numeric_features = (
        "loan_amount_000s", "applicant_income_000s",
        "population", "minority_population", "hud_median_family_income",
        "tract_to_msamd_income", "number_of_owner_occupied_units",
        "number_of_1_to_4_family_units",
    )
    categorical_features = (
        "loan_type", "property_type", "loan_purpose", "owner_occupancy",
        "lien_status", "state_code",
    )

    APPROVED = frozenset({"1", "2", "8"})
    DENIED = frozenset({"3", "7"})

    # The domain classifier's stated minimum. Below 200 the domain classifier does not
    # run, and it is mandatory for alert attribution -- a corpus built under that
    # threshold would carry no multivariate attribution on any window.
    ROWS_PER_WINDOW = 200
    WINDOW_SECONDS = 300                      # default emission cadence

    def __init__(self, *, data_dir: Path | None = None,
                 rows_per_year: int = 2_000_000, seed: int = 7) -> None:
        self._data_dir = Path(data_dir) if data_dir else None
        self.rows_per_year = rows_per_year
        self.seed = seed

    def _dir(self) -> Path:
        import os
        if self._data_dir:
            if not self._data_dir.exists():
                raise FileNotFoundError(
                    f"{self.dataset_id}: no data at {self._data_dir}\n  "
                    f"{self.acquisition_note}")
            return self._data_dir
        env = os.getenv("HMDA_DATA_DIR")
        if env:
            self._data_dir = Path(env)
            return self._dir()
        return self._require()

    def load(self, target: str = "approval") -> Split:  # pragma: no cover - needs real data
        raise NotImplementedError(
            "HmdaAdapter.load is implemented in the corpus build path, which "
            "streams the gzipped years rather than materialising 200M rows. "
            "See corpus/hmda_stream.py.")


REGISTRY: dict[str, type] = {
    "synthetic": SyntheticAdapter,
    "folktables": FolktablesAdapter,
    "backblaze": BackblazeAdapter,
    "hmda": HmdaAdapter,
}


def get_adapter(dataset_id: str, **kwargs):
    if dataset_id not in REGISTRY:
        raise KeyError(f"unknown dataset '{dataset_id}'; known: {sorted(REGISTRY)}")
    return REGISTRY[dataset_id](**kwargs)


class NasaPCoEAdapter(_RealDataAdapter):
    """NASA Prognostics Center of Excellence run-to-failure repository.

    21 files clustering into 14 INDEPENDENT physical domains.
    Five battery datasets share failure physics and are no more independent
    than five learning rates -- counting files rather than domains would
    inflate `G` by 105, an error that has been caught three times.

    Terms (retrieved 2026-08-11): NASA does not licence its materials;
    attribution is REQUESTED, not imposed. No redistribution or commercial
    restriction stated. US Government work.
    train_commercial + redistribute + demo all permitted.

    This is the corpus's answer to the censored time-to-event gap. Nearly every
    dataset here is run-to-failure, so units surviving the observation window
    are RIGHT-CENSORED -- usable signal for the forecaster's survival head, not rows
    to drop. Before this, Backblaze was the only source and the survival head
    would have learned censoring entirely from hard-drive failure.
    """

    dataset_id = "nasa_pcoe"
    expected_path = "nasa_pcoe"
    acquisition_note = (
        "Download from the NASA PCoE repository. Cite both the repository and "
        "the data donators -- attribution is the one thing NASA asks for. "
        "Do not commit raw files; fetch at build time."
    )

    # 14 independent domains, not 21 files. Each maps to one `group_id` prefix.
    DOMAINS = (
        "turbofan_cmapss", "turbofan_cmapss2", "turbofan_phm08",
        "bearings_ims", "bearings_femto",
        "battery_randomized", "battery_accelerated",
        "semiconductor_igbt", "semiconductor_mosfet",
        "machining_milling", "machining_cnc",
        "structural_cfrp", "structural_fatigue_crack",
        "misc_anemometer",
    )

    targets = ("remaining_useful_life",)
    numeric_features = ()      # per-domain; each carries its own sensor set
    categorical_features = ()

    def load(self, target: str) -> Split:  # pragma: no cover - needs real data
        self._require()
        raise NotImplementedError(
            "Per-domain sensor schemas differ (turbofan has 21 channels, "
            "bearings vibration spectra, batteries charge/discharge cycles), so "
            "each domain needs its own feature mapping. Implement per domain, "
            "and PRESERVE right-censoring: units surviving the observation "
            "window are signal for survival modelling, not rows to drop.")


class FhfaPudbAdapter(_RealDataAdapter):
    """FHFA Enterprise Public Use Database -- the regime-break candidate.

    Annual loan-level releases 2008-present, mandated by federal law to
    disclose Fannie Mae and Freddie Mac mortgage data. Covers the crisis
    period directly, which is the shift class the corpus otherwise cannot span.

    *** THE CATCH, AND IT IS NOT SMALL ***

    PUDB carries ORIGINATION characteristics, not loan performance history.
    There is no default outcome, so it CANNOT serve default prediction.

    It is still usable on a CONSTRUCTED target -- predict an origination
    characteristic (affordability band, LTV band) from the others. The 2008
    break is a genuine, large shift in the applicant and origination space,
    which is what the corpus needs: a base model degrading against real regime
    change. But the target is ours, not the data's, and that must stay visible
    wherever this dataset's results are reported.

    Licence terms are NOT stated on the FHFA page. Likely public domain as US
    federal work; confirmation requested from FHFA 2026-08-11.
    Until it returns, this source is PROVISIONAL and must not enter a shipped
    corpus.
    """

    dataset_id = "fhfa_pudb"
    expected_path = "fhfa_pudb"
    acquisition_note = (
        "Download annual single-family and multifamily PUDB files from FHFA. "
        "LICENCE UNCONFIRMED -- terms not stated on the page; confirmation "
        "requested 2026-08-11. Do not include in a shipped corpus until it "
        "returns."
    )
    targets = ("affordability_band", "ltv_band")
    numeric_features = ("borrower_income", "ltv", "note_age", "unpaid_balance")
    categorical_features = ("census_tract", "purpose", "seller_type")

    def load(self, target: str) -> Split:  # pragma: no cover - needs real data
        self._require()
        raise NotImplementedError(
            "Requires a CONSTRUCTED target -- PUDB has no performance outcome. "
            "Predict an origination characteristic from the others, and record "
            "in provenance that the target is constructed so no reader mistakes "
            "this for default prediction.")


REGISTRY["nasa_pcoe"] = NasaPCoEAdapter
REGISTRY["fhfa_pudb"] = FhfaPudbAdapter
