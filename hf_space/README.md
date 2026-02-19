---
title: VERITAS Deepfake Detector
emoji: 🔍
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# VERITAS on Hugging Face Spaces

This Space runs VERITAS deepfake face detection using EfficientNet-B0 + EXIF metadata heuristics.

## Included files
- `app.py`: Gradio UI entrypoint
- `predict.py`: EfficientNet-B0 inference
- `metadata.py`: EXIF-based metadata analysis
- `model.pth`: trained weights

## Notes
- Space runtime installs dependencies from `requirements.txt`.
- If the Space fails at startup, check logs for missing model or dependency issues.
