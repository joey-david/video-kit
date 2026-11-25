#!/usr/bin/env bash
# Recreate the video_kit environment (no sudo). System deps expected from Homebrew/MacTeX:
#   ffmpeg espeak-ng imagemagick poppler; MacTeX (latex, dvisvgm); fonts CMU Serif, JetBrains Mono, STIX Two.
set -euo pipefail
cd "$(dirname "$0")"
for c in ffmpeg espeak-ng magick latex dvisvgm uv; do command -v "$c" >/dev/null || echo "WARNING: $c not found"; done
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python manim "kokoro>=0.9" soundfile pymupdf requests parakeet-mlx num2words \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
bin/vk say "Setup complete. The narrator is ready." >/dev/null && echo "kokoro OK"
echo "done. Try: bin/vk new paper.pdf"
