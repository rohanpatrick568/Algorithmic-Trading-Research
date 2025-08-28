#!/usr/bin/env bash
set -euo pipefail

# Config
VENV_DIR="${1:-venv}"   # default to ./venv (ignored by .gitignore)

# Resolve a Python 3 interpreter
pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    # Verify it's Python 3
    if python -c "import sys; sys.exit(0 if sys.version_info[0]==3 else 1)" 2>/dev/null; then
      echo "python"
    else
      echo ""
    fi
  elif command -v py >/dev/null 2>&1; then
    echo "py -3"
  else
    echo ""
  fi
}

PYTHON_BIN="$(pick_python)"
if [ -z "${PYTHON_BIN}" ]; then
  echo "Error: Python 3 not found. Please install Python 3.8+ and retry." >&2
  exit 1
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Creating virtual environment in: ${VENV_DIR}"
${PYTHON_BIN} -m venv "${VENV_DIR}"

# Activate (Windows vs POSIX)
if [ -f "${VENV_DIR}/Scripts/activate" ]; then
  # Windows (cmd/PowerShell/Git-Bash)
  # shellcheck disable=SC1091
  source "${VENV_DIR}/Scripts/activate"
else
  # POSIX (Linux/Mac)
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

# Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# Install dependencies if present
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  python -m pip install -r requirements.txt
else
  echo "No requirements.txt found; skipping dependency install."
fi

echo
echo "Environment ready."
echo "To activate later:"
if [ -f "${VENV_DIR}/Scripts/activate" ]; then
  echo "  source ${VENV_DIR}/Scripts/activate"
else
  echo "  source ${VENV_DIR}/bin/activate"
fi