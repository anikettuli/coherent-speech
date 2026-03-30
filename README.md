# Coherant Speech Restorer

Coherant Speech is a high-performance, AI-driven video restoration pipeline. It revitalizes videos with poor-quality audio by generating clean, studio-quality speech using state-of-the-art Natural Language Processing (NLP) and Text-to-Speech (TTS) technologies, and aligns it perfectly with the original video timeline.

## Features

- **High-Fidelity Voice Generation**: Leveraging the bleeding-edge Kokoro-82M model for expressive, built-in studio voices.
- **Zero-Shot Voice Cloning**: Includes experimental support for zero-shot voice cloning using F5-TTS. It can automatically extract voice DNA from the source video to synthesize speech that sounds like the original speaker.
- **Robust Automatic Speech Recognition (ASR)**: Uses Faster-Whisper to transcribe audio accurately. Includes a caching mechanism to avoid re-running expensive transcriptions.
- **Dynamic Hardware Acceleration**: Automatically detects and utilizes NVIDIA GPUs (CUDA) and Apple Silicon M-Series chips (MPS) for maximum inference speed, with robust multithreaded CPU fallback mechanisms for maximum compatibility across devices.
- **Modern User Interface**: A responsive and interactive visual playground built with Streamlit, enabling intuitive file uploads, hardware selection, and real-time inference progress tracking.
- **Cross-Platform Speed**: Built to take advantage of native Python speeds, managed by the blazingly fast `uv` package manager.

## Prerequisites

- **Python:** Highly compatible with native Python 3.13.
- **System Packages:**
  - `ffmpeg`: Required for audio extraction and video assembly.
  - `espeak-ng`: System phonemizer required for the Kokoro TTS engine.

On debian/ubuntu machines, you can install the system requirements via:
```sh
sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng
```

## Setup & Installation

The project uses `uv`, an extremely fast Python package manager.

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd coherant-speech
   ```

2. **Automated Setup:**
   You can run the provided interactive script to install dependencies, set up the virtual environment, and launch the application all in one go.
   ```sh
   ./run.sh
   ```

3. **Manual Setup (Optional):**
   If you aren't using `run.sh`, you can set up the environment manually with `uv`:
   ```sh
   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create and activate virtual environment
   uv venv venv
   source venv/bin/activate
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

## Usage

If you've installed manually, you can start the development server using Streamlit:
```sh
streamlit run streamlit_app.py
```
*(Note: Be sure your LD_LIBRARY_PATH is configured correctly for CUDA if you are taking advantage of GPU acceleration, as handled inside `run.sh`).*

1. **Upload Source:** Upload your target lecture video (mp4, mkv, mov).
2. **Configure Pipeline:** Select hardware acceleration, Whisper target precision, and the desired Voice Synthesis Strategy (Studio Voices or Voice Cloning).
3. **Execute:** Click "Execute Restoration" to begin processing. You can stop the pipeline halfway gracefully using the visual Stop button.
4. **Download:** Play the fully restored video directly in your browser, or download it.

## Architecture

- `streamlit_app.py` / `app.py`: The user interface and visual configurations. Built with Streamlit and Gradio (legacy).
- `pipeline.py`: The core audio/video multiplexing logic. Orchestrates ffmpeg extractions, ASR, TTS inference alignment, and timeline matching.
- `tts_manager.py`: Abstraction layer managing interactions with TTS inferencing engines (Kokoro and F5), including hardware dispatching and memory safe threading operations.
- `run.sh`: Automated application launcher ensuring library bindings and prerequisite package resolutions.
