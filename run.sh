#!/bin/bash
set -e

echo "Setting up Modernized Coherant Speech App Environment (2025)..."

if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! dpkg -l | grep -q espeak-ng; then
    echo "Installing espeak-ng (System Phonemizer required for Kokoro TTS)..."
    sudo apt-get update && sudo apt-get install espeak-ng -y
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment with native Python (3.13 Supported!)..."
    uv venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing modern dependencies blazingly fast..."
uv pip install -r requirements.txt

echo "Starting Application..."
# Patch LD_LIBRARY_PATH so CUDA libs (cublas, cudnn, npp) can be found
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
for d in "$SITE_PACKAGES"/nvidia/*/lib; do
    if [ -d "$d" ]; then
        export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
    fi
done

streamlit run streamlit_app.py

