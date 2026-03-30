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
python app.py
