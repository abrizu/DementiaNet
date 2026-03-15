#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

# ── Python version check (>= 3.9) ────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "Error: Python not found. Install Python 3.9 or newer."
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    echo "Error: Python 3.9+ required (found $PY_VER)."
    exit 1
fi
echo "Python $PY_VER detected."

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── Install dependencies ──────────────────────────────────────────────────────
echo "Installing Python dependencies ..."
pip install --upgrade pip --quiet
pip install --quiet \
    torch torchvision torchaudio \
    numpy pandas scikit-learn scipy matplotlib \
    kaggle nibabel tqdm requests \
    streamlit plotly librosa soundfile Pillow

# ── Launch app ────────────────────────────────────────────────────────────────
echo ""
echo "Starting DementiaNet on http://localhost:8501"
echo ""
streamlit run "$REPO_DIR/app.py" --server.port 8501
