# BlurredBox Edge Agent

The BlurredBox Edge Agent is a small program that you run on your own machine, next to your machine learning model. It watches how your model behaves and works out short summaries of that behaviour on your side.

Your raw data stays with you. The agent runs on your infrastructure and sends BlurredBox only per-window summary statistics: score quantiles (the 1st to 99th percentile of the model's outputs for the window) and per-feature drift statistics under hashed feature ids. Individual records, raw feature values, feature names, logs and prices are not sent. A model's score distribution is, however, what the quantiles describe, so treat the summaries as derived from your predictions, not as unrelated to them.

## What leaves your network

- Signed HTTPS posts of the summary statistics, to the endpoints in your config (`/api/v1/ingest/signal`).
- Optional, off by default: if you set `instrument_series`, the agent downloads public daily closes from `fred.stlouisfed.org` (one request per series per refresh). That tells FRED which series you follow.

Nothing else is sent. The code that assembles a payload is `edge_agent/wire.py`; `tests/edge_agent/test_wire_audit.py` checks it for leaks.

## Platform

Linux (systemd unit or the container image) is the supported platform.

Windows is not supported. `packaging/edge_agent/installer/windows/` ships an installer (`install.ps1`) and a service wrapper that sends CTRL_BREAK on stop so the open window is kept, and the installer restricts `conf\` and `data\` with ACLs. None of it has been run on Windows. In the agent code itself the `0o600` file modes have no effect on Windows; only the installer's ACLs protect the state files, so a manual run there leaves them readable by other local users. Use it only for your own testing.

## Status

Early stage. This repository mirrors the edge agent from BlurredBox `dev` at `838946898` (after release `v0.0.2-rc6`, before rc7; it carries the 2026-11-04 audit fixes) with `scripts/release_from_agd.py`. The agent needs the BlurredBox SDK (`blurredbox_agd`), which is not on a public index; without it, and without the server code some tests import, those tests skip. To report a security problem see `SECURITY.md`.

## Install

Use the install kit you receive from BlurredBox; it installs offline with `--no-index` and does not use a public package index. Installer and service files are in `packaging/edge_agent/`. Configuration is a JSON file (`python -m edge_agent --config agent.json`); the HMAC secret and BlurredBox API key come from the environment (`AGD_HMAC_SECRET`, `BLURREDBOX_API_KEY`) or the hidden prompt of `blurredbox-edge setup`. `blurredbox-edge check` inspects a sample of your input without sending anything.

## Licence

Source-available under the BlurredBox Edge Agent License 1.0 (see `LICENSE`). Use in AI or machine-learning governance, in any financial activity, or in a competing product is not permitted without written permission.
