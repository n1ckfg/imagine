---
title: Imagine
emoji: 🎨
colorFrom: gray
colorTo: gray
sdk: static
app_build_command: npm run build
app_file: build/index.html
pinned: false
license: mit
models:
  - lox9973/MiniT2I-B-16-MF-compact
  - cfahlgren1/MiniT2I-B-16-MF-compact-webgpu-int8
short_description: Type anything and watch it paint, on your own GPU
---

# Imagine

Type anything and watch it paint. Each 512 × 512 image is made in your browser on your own GPU with WebGPU, in about half a second on a recent laptop.

The model is [MiniT2I-B-16-MF-compact](https://huggingface.co/lox9973/MiniT2I-B-16-MF-compact) by lox9973, a compact fine-tune of [MiniT2I](https://peppaking8.github.io/minit2i/), exported to ONNX in [cfahlgren1/MiniT2I-B-16-MF-compact-webgpu-int8](https://huggingface.co/cfahlgren1/MiniT2I-B-16-MF-compact-webgpu-int8).
