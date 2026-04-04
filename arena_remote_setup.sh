#!/usr/bin/env bash
# setup_arena_remote.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CONDA_ENV_NAME="arena-env"
# ─────────────────────────────────────────────────────────────────────────────

conda env create -f ~/arena/environment.yml -n "${CONDA_ENV_NAME}" || \
conda env update -f ~/arena/environment.yml -n "${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"
pip install -r ~/arena/pip-reqs.txt
