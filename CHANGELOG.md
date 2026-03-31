# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-31
### Added
- Complete rebranding to `Coherent Speech Studio`.
- Enterprise-grade AI audio restoration capabilities.
- Dynamic hardware allocation (`auto`, `cuda`, `mps`, `cpu`) using `os.cpu_count()`.
- Thread-safe TTS generation and `F5-TTS` inference path monkey-patches.
- Support for `requirements-gpu.txt` isolation for portability.
- Collision-free Streamlit `TemporaryDirectory` isolation for web app users.
- Smoke tests using `pytest`.
- Fully documented modern `README.md` containing professional use cases and shields flags.

### Removed
- Legacy dead `app.py` Gradio UI.
- Hardcoded string path assignments, ensuring seamless multithreading.
