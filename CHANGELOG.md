# Changelog

All notable changes to this project, one line per phase. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); commits follow Conventional Commits.

## [Unreleased]

- Phase 0: project scaffold — pyproject (Python 3.12), ruff, pydantic-settings config, module tree, Makefile, DVC init, test framework (6 tests pass).
  - GPU verified: RTX 5060 Ti (Blackwell sm_120) runs torch 2.12.0+cu130; GPU matmul confirmed → DeepAR will train on GPU at full 50 epochs (no CPU fallback needed).
- Phase 1: data pipeline + EDA — download (UCI #501 → synthetic fallback), preprocess (continuous hourly index, gap-aware imputation, backward-only features), strict temporal split, pandera schema, DVC pipeline (download→preprocess→split), EDA notebook (5 plots). 22 tests pass.
  - **Real UCI data used** (`SOURCE=uci`): station Aotizhongxin, 35064 hourly rows, raw PM2.5 missingness 2.64%. Splits: train 29232 / val 4416 / test 1416. Long gaps (>6h) left NaN and excluded from eval.
