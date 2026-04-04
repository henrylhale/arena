#!/usr/bin/env bash
# arena_export_env.sh — export the current arena-env state to environment.yml and pip-reqs.txt

set -euo pipefail

CONDA_ENV_NAME="arena-env"
REPO_ROOT="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"

# Ensure the environment exists
if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  echo "Error: conda environment '${CONDA_ENV_NAME}' not found" >&2
  exit 1
fi

conda env export -n "${CONDA_ENV_NAME}" > "${REPO_ROOT}/environment.yml"
conda run -n "${CONDA_ENV_NAME}" pip freeze > "${REPO_ROOT}/pip-reqs.txt"

echo "Exported ${CONDA_ENV_NAME} to:"
echo "  ${REPO_ROOT}/environment.yml"
echo "  ${REPO_ROOT}/pip-reqs.txt"
