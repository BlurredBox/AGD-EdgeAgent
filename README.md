# BlurredBox Edge Agent

The BlurredBox Edge Agent is a small program that you run on your own machine, next to your machine learning model. It watches how your model behaves and works out short summaries of that behaviour on your side.

Your raw data stays with you. The agent runs on your infrastructure, and only the summaries it produces are sent to the BlurredBox service. Your inputs, predictions and other raw records never leave your environment.

## Status

Early stage and under active development. This repository mirrors the edge agent from BlurredBox release `v0.0.2-rc6` (agent version 0.1.3). Some modules and tests import server code that is not in this repository, so not every test runs from a clean clone.

## Install

Use the install kit you receive from BlurredBox; it installs offline with `--no-index` and does not use a public package index. Installer and service files are in `packaging/edge_agent/`.

## Licence

Source-available under the BlurredBox Edge Agent License 1.0 (see `LICENSE`). Use in AI or machine-learning governance, in any financial activity, or in a competing product is not permitted without written permission.
