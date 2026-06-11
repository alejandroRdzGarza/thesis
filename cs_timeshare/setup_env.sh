#!/bin/bash
# Run this ONCE on cream to create the Python virtual environment.
# SSH into cream first, then run:
#   bash ~/thesis/cs_timeshare/setup_env.sh

set -e

# UCL CS machines require CS_OS to be set; vanilla's login shell doesn't always set it.
export CS_OS="${CS_OS:-linux}"

THESIS_DIR="/cs/student/project_msc/2025/rai/jesusr01/thesis"
VENV_DIR="${THESIS_DIR}/venv"
REQUIREMENTS="${THESIS_DIR}/VLA-Model/openvla/requirements_server.txt"
export PIP_CACHE_DIR="${THESIS_DIR}/.pip_cache"

echo "=== Setting up Python 3.11 ==="
if [[ -f /opt/Python/Python-3.11.5_Setup.sh ]]; then
    source /opt/Python/Python-3.11.5_Setup.sh
else
    export PATH="/opt/Python/Python-3.11.5/bin:$PATH"
fi

echo "Python: $(python3 --version)"

echo "=== Creating virtual environment at $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

echo "=== Installing dependencies ==="
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r "$REQUIREMENTS"
pip install bitsandbytes>=0.42.0

echo ""
echo "=== Setup complete. Venv: $VENV_DIR ==="
echo ""
echo "Next: start the server with:"
echo "  tmux new -s vla"
echo "  bash ${THESIS_DIR}/cs_timeshare/run_server.sh"
