"""
corpus/multivariate.py

Five multivariate detectors — the failure mode univariate
tests miss entirely: a correlation-structure break where every marginal is
unchanged.

*** THE DOMAIN CLASSIFIER IS INVARIANT-MANDATED ***

Four of these five return a single scalar for the whole vector. Only the
domain classifier attributes per feature, so for a correlation break it is the
ONLY thing that can name a feature. The alert-attribution rule forbids an unattributable alert, so
it is excluded from the load-shedding order entirely.

Until this module existed, that mandated capability was specified and absent.

Uses numpy/sklearn: these run at the edge where the raw feature values are,
and the edge tier already carries a numerical stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["MultivariateReference", "pca_reconstruction_error",
           "domain_classifier", "mmd_rbf", "mahalanobis_mean",
           "covariance_delta", "top_eigenvalue_ratio", "DomainClassifierResult"]

# MMD is O(n^2). Capped per side so cost is constant per window
# rather than quadratic in window size. Field present every window either way.
MMD_SUBSAMPLE_CAP = 500

# The classifier trains per window, so the reference side is
# subsampled to the current window's size -- otherwise it learns "predict
# reference" and the AUC is meaningless.
DOMAIN_CLF_MIN_ROWS = 200   # stated minimum


@dataclass
class MultivariateReference:
    """Reference-side state, fitted ONCE per reference window.

    Everything here is frozen for the reference window's lifetime. Refitting
    per emission window would make every statistic compare against a moving
    target -- the same error the univariate path avoids by caching bin edges.
    """
    mean: np.ndarray
    std: np.ndarray
    pca_components: np.ndarray          # (k, p)
    pca_inbase_error: float             # reference's own reconstruction error
    precision: np.ndarray               # inverse Ledoit-Wolf covariance
    mahalanobis_median: float           # for novelty normalisation
    correlation: np.ndarray
    top_eigenvalue: float
    mmd_bandwidth: float                # median heuristic, frozen
    sample: np.ndarray                  # standardised reference sample
    n_features: int

    @classmethod
    def fit(cls, X: np.ndarray, *, variance_target: float = 0.95,
            rng_seed: int = 0) -> "MultivariateReference":
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0             # constant feature: do not divide by 0
        Z = (X - mean) / std

        # --- PCA, components frozen at fit time
        cov = np.cov(Z, rowvar=False)
        cov = np.atleast_2d(cov)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        total = eigvals.sum()
        if total <= 0:
            k = 1
        else:
            cumulative = np.cumsum(eigvals) / total
            k = int(np.searchsorted(cumulative, variance_target) + 1)
        k = max(2, min(k, max(2, min(20, p // 2 if p >= 4 else p))))
        k = min(k, p)
        comps = eigvecs[:, :k].T

        recon = (Z @ comps.T) @ comps
        inbase = float(np.mean(np.sum((Z - recon) ** 2, axis=1)))

        # --- Ledoit-Wolf shrinkage
        precision = _ledoit_wolf_precision(Z)
        d = _mahalanobis(Z, precision)
        maha_median = float(np.median(d)) or 1.0

        corr = _correlation(Z)
        top_eig = float(np.linalg.eigvalsh(corr)[-1])

        bandwidth = _median_heuristic(Z, rng_seed)

        return cls(mean=mean, std=std, pca_components=comps,
                   pca_inbase_error=inbase, precision=precision,
                   mahalanobis_median=maha_median, correlation=corr,
                   top_eigenvalue=top_eig, mmd_bandwidth=bandwidth,
                   sample=Z, n_features=p)

    def standardise(self, X: np.ndarray) -> np.ndarray:
        """Always with REFERENCE mean/std. Per-window standardisation would
        make every statistic measure scaling rather than structure."""
        return (np.asarray(X, dtype=float) - self.mean) / self.std


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ledoit_wolf_precision(Z: np.ndarray) -> np.ndarray:
    """Inverse covariance under Ledoit-Wolf shrinkage toward a scaled identity.

    Financial features are correlated (near-singular covariance,
    so inversion amplifies noise) and heavy-tailed (a few rows dominate the
    sample covariance). Shrinkage fixes the first directly and the second
    partially; MCD would do better on the second but is far more expensive and
    itself unstable in high dimension.
    """
    n, p = Z.shape
    S = np.cov(Z, rowvar=False)
    S = np.atleast_2d(S)
    mu = np.trace(S) / p
    target = mu * np.eye(p)

    d2 = np.sum((S - target) ** 2)
    # Vectorised. The obvious loop -- build x_i x_i^T for each row and sum the
    # squared Frobenius difference from S -- is O(n) in PYTHON and unusable at
    # the reference-window sizes the contract requires (>= 2000 rows AND >= 24h,
    # which at realistic window sizes means tens of thousands of rows).
    #
    #   ||x x^T - S||_F^2  =  (x.x)^2  -  2 x^T S x  +  ||S||_F^2
    #
    # All three terms are computable for every row at once.
    sq_norms = np.einsum("ij,ij->i", Z, Z)              # (x.x) per row
    quad = np.einsum("ij,jk,ik->i", Z, S, Z)            # x^T S x per row
    frob_S = np.sum(S ** 2)
    b2 = float(np.sum(sq_norms ** 2 - 2 * quad + frob_S))
    b2 = min(d2, b2 / (n ** 2)) if n else 0.0
    shrink = (b2 / d2) if d2 > 0 else 0.0
    Sigma = shrink * target + (1 - shrink) * S
    # ridge only if still singular -- shrinkage should already have fixed it
    try:
        return np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Sigma + 1e-8 * np.eye(p))


def _mahalanobis(Z: np.ndarray, precision: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, precision, Z), 0.0))


def _correlation(Z: np.ndarray) -> np.ndarray:
    C = np.corrcoef(Z, rowvar=False)
    return np.atleast_2d(np.nan_to_num(C, nan=0.0))


def _median_heuristic(Z: np.ndarray, seed: int, cap: int = 300) -> float:
    """sigma^2 = median pairwise squared distance, from the REFERENCE only,
    computed once and frozen. Recomputing per window makes the bandwidth move
    with the data and the statistic stops being comparable across time."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(cap, len(Z)), replace=False)
    S = Z[idx]
    diff = S[:, None, :] - S[None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    iu = np.triu_indices(len(S), k=1)
    med = float(np.median(d2[iu])) if len(iu[0]) else 1.0
    return med if med > 0 else 1.0


# ---------------------------------------------------------------------------
# The five detectors
# ---------------------------------------------------------------------------

def pca_reconstruction_error(ref: MultivariateReference,
                             X: np.ndarray) -> float:
    """Ratio of current reconstruction error to the reference's own.

    The RATIO is the useful number -- raw error scales with feature count and
    is not comparable across monitored models.
    """
    Z = ref.standardise(X)
    recon = (Z @ ref.pca_components.T) @ ref.pca_components
    err = float(np.mean(np.sum((Z - recon) ** 2, axis=1)))
    return err / ref.pca_inbase_error if ref.pca_inbase_error > 0 else 0.0


@dataclass
class DomainClassifierResult:
    """drift_score plus PER-FEATURE attribution -- the reason this detector is
    invariant-mandated."""
    drift_score: float                  # 2*AUC - 1, in [0, 1]
    auc: float
    attribution: dict[int, float]       # feature index -> normalised importance


def domain_classifier(ref: MultivariateReference, X: np.ndarray, *,
                      seed: int = 0) -> DomainClassifierResult | None:
    """Reference-vs-current discriminator. Returns None below the minimum.

    Gradient-boosted trees rather than logistic regression: a LINEAR
    discriminator cannot separate distributions that differ in INTERACTION
    structure while sharing marginals -- which is the exact failure mode this
    whole module exists to cover. A linear domain classifier would be blind to
    the thing it is here to detect.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    Z_cur = ref.standardise(X)
    n_cur = len(Z_cur)
    if n_cur < DOMAIN_CLF_MIN_ROWS:
        return None

    rng = np.random.default_rng(seed)
    n_ref = min(len(ref.sample), n_cur)          # balance, or AUC is meaningless
    idx = rng.choice(len(ref.sample), size=n_ref, replace=False)
    Z_ref = ref.sample[idx]

    Xall = np.vstack([Z_ref, Z_cur])
    y = np.concatenate([np.zeros(len(Z_ref)), np.ones(n_cur)])

    clf = HistGradientBoostingClassifier(max_iter=60, random_state=seed)
    # Cross-validated: AUC on training data is ~1.0 always and would report
    # permanent drift.
    proba = cross_val_predict(clf, Xall, y, cv=3, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    drift = max(0.0, 2 * auc - 1)

    # Per-feature attribution via permutation importance on the fitted model.
    clf.fit(Xall, y)
    base = float(roc_auc_score(y, clf.predict_proba(Xall)[:, 1]))
    importances = np.zeros(Xall.shape[1])
    for j in range(Xall.shape[1]):
        Xp = Xall.copy()
        rng.shuffle(Xp[:, j])
        permuted = float(roc_auc_score(y, clf.predict_proba(Xp)[:, 1]))
        importances[j] = max(0.0, base - permuted)
    total = importances.sum()
    attribution = ({j: float(importances[j] / total) for j in range(len(importances))}
                   if total > 0 else {j: 0.0 for j in range(len(importances))})

    return DomainClassifierResult(drift_score=drift, auc=auc,
                                  attribution=attribution)


def mmd_rbf(ref: MultivariateReference, X: np.ndarray, *,
            seed: int = 0, cap: int = MMD_SUBSAMPLE_CAP) -> float:
    """Unbiased MMD^2 with an RBF kernel at the frozen reference bandwidth.

    Subsampled to `cap` per side so cost is constant per window rather than
    quadratic in window size. Seeded, so a replay is reproducible.
    """
    rng = np.random.default_rng(seed)
    Z_cur = ref.standardise(X)
    A = ref.sample
    if len(A) > cap:
        A = A[rng.choice(len(A), size=cap, replace=False)]
    if len(Z_cur) > cap:
        Z_cur = Z_cur[rng.choice(len(Z_cur), size=cap, replace=False)]

    g = 1.0 / ref.mmd_bandwidth

    def k(P, Q):
        d2 = np.sum((P[:, None, :] - Q[None, :, :]) ** 2, axis=-1)
        return np.exp(-g * d2)

    m, n = len(A), len(Z_cur)
    if m < 2 or n < 2:
        return 0.0
    Kaa, Kbb, Kab = k(A, A), k(Z_cur, Z_cur), k(A, Z_cur)
    np.fill_diagonal(Kaa, 0.0)          # unbiased: exclude diagonal
    np.fill_diagonal(Kbb, 0.0)
    return float(Kaa.sum() / (m * (m - 1))
                 + Kbb.sum() / (n * (n - 1))
                 - 2 * Kab.mean())


def mahalanobis_mean(ref: MultivariateReference, X: np.ndarray) -> float:
    """Mean Mahalanobis distance under the frozen reference precision."""
    return float(np.mean(_mahalanobis(ref.standardise(X), ref.precision)))


def covariance_delta(ref: MultivariateReference, X: np.ndarray) -> float:
    """Frobenius norm of the CORRELATION-matrix difference, normalised.

    Correlation rather than covariance so the statistic measures structure
    rather than scale -- scale movement is already the univariate detectors' job. Divided by
    sqrt(p(p-1)/2), the number of free off-diagonal entries, so it is
    comparable across models with different feature counts.
    """
    Z = ref.standardise(X)
    if len(Z) < 2:
        return 0.0
    delta = _correlation(Z) - ref.correlation
    p = ref.n_features
    denom = math.sqrt(p * (p - 1) / 2) if p > 1 else 1.0
    iu = np.triu_indices(p, k=1)
    return float(np.sqrt(np.sum(delta[iu] ** 2)) / denom)


def top_eigenvalue_ratio(ref: MultivariateReference, X: np.ndarray) -> float:
    """lambda1_current / lambda1_reference.

    lambda1 is dominant common-factor strength. A rising ratio means features
    are becoming more correlated -- in finance, the signature of a stress
    regime where everything starts moving together.
    """
    Z = ref.standardise(X)
    if len(Z) < 2:
        return 1.0
    top = float(np.linalg.eigvalsh(_correlation(Z))[-1])
    return top / ref.top_eigenvalue if ref.top_eigenvalue > 0 else 1.0
