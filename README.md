---
title: Instruct Particulate
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
---

# Instruct Particulate Demo

This repository contains a self-contained Gradio demo for the
`FINAL-ours-final_2026-05-01T12-36-12` Instruct Particulate checkpoint.
On Hugging Face Spaces, the large model checkpoint is downloaded at startup
from `rayli/instruct-particulate-model`.

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python app.py
```

The app accepts one `.glb` or `.obj` mesh and a kinematic tree JSON. Link IDs
must be dense integers starting at `0`; joints use `revolute` or `prismatic`.
