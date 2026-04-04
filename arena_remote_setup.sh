#!/usr/bin/env bash
# arena_remote_setup.sh — set up the arena-env conda environment

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CONDA_ENV_NAME="arena-env"
REPO_ROOT="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
# ─────────────────────────────────────────────────────────────────────────────

conda env create -f "${REPO_ROOT}/environment.yml" -n "${CONDA_ENV_NAME}" || \
conda env update -f "${REPO_ROOT}/environment.yml" -n "${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"
pip install -r "${REPO_ROOT}/pip-reqs.txt"
