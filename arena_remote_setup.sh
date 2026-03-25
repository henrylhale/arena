#!/usr/bin/env bash
# setup_arena_remote.sh
# Run once on a fresh Vast.ai instance to get a working ARENA environment.
# Swap in your actual GitHub usernames/repo names before using.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_USER="henrylhale"
DOTFILES_REPO="configs"        # e.g. "dotfiles"
ARENA_REPO="arena"         # e.g. "ARENA_3.0"
CONDA_ENV_NAME="arena-env"
NVIM_VERSION="0.11.6"                     # pin to a known-good release
# ─────────────────────────────────────────────────────────────────────────────

echo "==> [1/5] Installing Neovim ${NVIM_VERSION}"
curl -fsSL \
  "https://github.com/neovim/neovim/releases/download/v${NVIM_VERSION}/nvim-linux-x86_64.tar.gz" \
  -o /tmp/nvim.tar.gz
tar -C /usr/local --strip-components=1 -xzf /tmp/nvim.tar.gz
rm /tmp/nvim.tar.gz
nvim --version | head -1
echo 'alias vim=nvim' >> ~/.bashrc

echo "==> [2/5] Pulling dotfiles (nvim, tmux)"
git clone --depth=1 \
  "https://github.com/${GITHUB_USER}/${DOTFILES_REPO}.git" \
  ~/configs

mkdir -p ~/.config
ln -sf ~/configs/nvim ~/.config/nvim
ln -sf ~/configs/.tmux.conf ~/.tmux.conf
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
tmux source-file ~/.tmux.conf 2>/dev/null || true
TMUX_PLUGIN_MANAGER_PATH=~/.tmux/plugins ~/.tmux/plugins/tpm/bin/install_plugins || true

echo "==> [3/5] Installing Neovim plugin manager + plugins (headless)"
# Assumes lazy.nvim — change the bootstrap path if you use a different manager.
nvim --headless "+Lazy! sync" +qa || true

echo "==> [4/5] Pulling ARENA repo"
git clone --depth=1 \
  "https://github.com/${GITHUB_USER}/${ARENA_REPO}.git" \
  ~/arena

echo "==> [5/6] Creating conda environment from environment.yml"
# The script assumes environment.yml is at the repo root.
# Vast.ai images may wrap conda with a shim that creates venvs under /venv/.
# We use it anyway and supplement with pip-reqs.txt for pip-only packages.
if ! command -v conda &>/dev/null; then
  echo "    conda not found — installing Miniconda"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p ~/miniconda3
  rm /tmp/miniconda.sh
  # Make conda available for the rest of this script
  source ~/miniconda3/etc/profile.d/conda.sh
  # Also wire it into .bashrc for future sessions
  ~/miniconda3/bin/conda init bash
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda env create -f ~/arena/environment.yml -n "${CONDA_ENV_NAME}" || \
  conda env update -f ~/arena/environment.yml -n "${CONDA_ENV_NAME}"

conda activate "${CONDA_ENV_NAME}"

echo "==> [6/6] Installing pip packages from pip-reqs.txt"
pip install -r ~/arena/pip-reqs.txt

echo ""
echo "==> Done. To activate the environment:"
echo "    conda activate ${CONDA_ENV_NAME}"
