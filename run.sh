#!/bin/bash
set -e

echo "Setting up Coherent Speech Studio Environment..."

if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! dpkg -l | grep -q espeak-ng 2>/dev/null; then
    echo "Installing espeak-ng (System Phonemizer required for Kokoro TTS)..."
    sudo apt-get update && sudo apt-get install espeak-ng -y
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    uv venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Syncing dependencies..."
uv pip install -r requirements.txt

# Check for GPU (NVIDIA) to install CUDA libs if requested
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU Detected. Installing CUDA acceleration libraries..."
    uv pip install -r requirements-gpu.txt
fi

echo "Starting Application..."
# Patch LD_LIBRARY_PATH so CUDA libs can be found natively
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
if [ -d "$SITE_PACKAGES/nvidia" ]; then
    for d in "$SITE_PACKAGES"/nvidia/*/lib; do
        if [ -d "$d" ]; then
            export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
        fi
    done
fi

streamlit run streamlit_app.py

