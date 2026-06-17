# Changelog

All notable changes to this project, one line per phase. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); commits follow Conventional Commits.

## [Unreleased]

- Phase 0: project scaffold — pyproject (Python 3.12), ruff, pydantic-settings config, module tree, Makefile, DVC init, test framework (6 tests pass).
  - GPU verified: RTX 5060 Ti (Blackwell sm_120) runs torch 2.12.0+cu130; GPU matmul confirmed → DeepAR will train on GPU at full 50 epochs (no CPU fallback needed).
