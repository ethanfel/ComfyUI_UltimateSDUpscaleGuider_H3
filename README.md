# ComfyUI_UltimateSDUpscaleGuider_H3

> **A fork of [ComfyUI_UltimateSDUpscaleGuider](https://github.com/Blakeem/ComfyUI_UltimateSDUpscaleGuider) with MiniMax H3 model support.**

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for running the image-to-image diffusion process on large images in tiles. Tiling improves the detail commonly lost on upscaled images while keeping VRAM use low and the working size close to what the diffusion model was trained on.

## Fork Changes

1. **MiniMax H3 model support**: original USDU nodes do not support MiniMax H3 model and produces errors when you try to connect it. This is a **vibe-coded** fork to bring H3 model support and use its potential for upscaling.

Please see example workflow: [minimax_h3_usdu.json](https://github.com/lisitskyaa/ComfyUI_UltimateSDUpscaleGuider_H3/blob/main/example_workflows/minimax_h3_usdu.json)

## Video memory handling

The Guider nodes assemble float image batches into one preallocated buffer,
release RGB tile buffers before sampling, and clear their shared image/model
references after success, failure, or cancellation. Pixel conversion and tile
sampling settings are unchanged.

Tiling is spatial: `batch_size=1` means one tile containing the whole clip, not
one video frame. Model weights, reference conditioning, temporal activations,
and the full input/output pixel clips still require memory. This is not a
temporal-windowed or disk-streaming upscaler.

CPU regression tests (no model downloads or GPU rendering):

```sh
COMFYUI_ROOT=/path/to/ComfyUI python test/memory_unit_test.py
```
