# ComfyUI_UltimateSDUpscaleGuider_H3

> **A fork of [ComfyUI_UltimateSDUpscaleGuider](https://github.com/Blakeem/ComfyUI_UltimateSDUpscaleGuider) with MiniMax H3 model support.**

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for running the image-to-image diffusion process on large images in tiles. Tiling improves the detail commonly lost on upscaled images while keeping VRAM use low and the working size close to what the diffusion model was trained on.

## Fork Changes

1. **MiniMax H3 model support**: original USDU nodes do not support MiniMax H3 model and produces errors when you try to connect it. This is a **vibe-coded** fork to bring H3 model support and use its potential for upscaling.

Please see example workflow: [minimax_h3_usdu.json](https://github.com/lisitskyaa/ComfyUI_UltimateSDUpscaleGuider_H3/blob/main/example_workflows/minimax_h3_usdu.json)

## VIDEO refinement and memory

`Ultimate SD Upscale (No Upscale, Guider, VIDEO)` accepts a file-backed native
ComfyUI `VIDEO`, such as Load Video or the output of the H3/DLSS5 VIDEO pipeline.
It inherits the IMAGE node's controls and uses the same tile and sampling engine.
The CAT pack is not required. Update both packs to make older workflows using
`CATUltimateSDUpscaleGuiderVideo` forward to this node automatically.

The VIDEO path decodes and encodes one frame at a time, avoiding the previous
adapter's full float32 input mmap and output tensor. Its optional `canvas_storage`
switch selects where the editable RGB canvas lives:

- `ram` (default): keep PIL frames in memory, preserving existing workflow behavior.
- `disk`: keep raw RGB frame files in temporary storage and load frames
  as needed. Use this for large clips under RAM pressure, with fast local scratch
  storage. The files require roughly `frames × width × height × 3` bytes and are
  removed after success, failure or cancellation. OS file-cache pages are
  reclaimable but can still appear as used memory.

`canvas_directory` optionally chooses the scratch folder for Disk mode. Leave
it empty to use ComfyUI's temp folder, or set a folder on a local SSD. A
RAM-backed filesystem such as tmpfs still holds these files in RAM, so choose
real disk storage to reduce system RAM pressure. RAM mode ignores this field.

Both settings use the same pixels, tile order, sampling and temporal context.
The storage switch does not change denoiser VRAM requirements. Both IMAGE and
VIDEO paths crop and resize one PIL tile frame at a time into the VAE buffer,
resize the canvas in place, and release shared buffers on every exit.

Output is temporary FFV1 RGB16 Matroska with source timestamps and stream-copied
audio. There is no intermediate CRF or YUV subsampling. USDU's existing internal
8-bit PIL pixel conversion remains unchanged; RGB16 transport does not add precision
to the sampled result. Final delivery encoding remains downstream. Lazy trimmed
or cropped VIDEO objects must first be saved and reloaded.

Tiling is spatial: `batch_size=1` means one tile containing the whole clip, not
one video frame. In RAM mode the PIL canvas still occupies memory; in both
modes the temporal float tile, model weights,
reference conditioning and sampling activations still require memory. This
change reduces RAM allocations; it does not introduce temporal chunking or
reduce the denoiser's VRAM requirement for the same settings. Smaller spatial
tiles/padding affect sampling memory; `tiled_decode` affects VAE decode memory.

CPU regressions (no model downloads or GPU rendering):

```sh
COMFYUI_ROOT=/path/to/ComfyUI python -B test/memory_unit_test.py
COMFYUI_ROOT=/path/to/ComfyUI python -B test/video_unit_test.py
python -B test/video_storage_unit_test.py
```

`test/benchmark_video_memory.py` compares process peak RSS and exact output pixel
hashes for IMAGE transport, the previous VIDEO adapter and native VIDEO in fresh
processes. Native `video` uses RAM; `video_disk` uses disk. Set
`USDU_VIDEO_BENCH_FRAMES` to choose the clip length (default 22). Its no-refinement run isolates transport memory, not H3 inference.

A CPU transport check on 22 frames at 1920×1088 (Python 3.13, Torch 2.11,
ComfyUI 0.34, no refinement) measured 2228 MiB peak process RSS for IMAGE,
2228 MiB for the previous mmap VIDEO adapter, and 1351 MiB for native VIDEO.
All three outputs had identical decoded pixel hashes. These figures include
about 803 MiB of imported runtime and are not full-workflow or GPU measurements.

With the same transport-only test extended to 175 frames at 1920×1088 using
local NVMe/ext4 scratch storage, native VIDEO measured 2595 MiB peak process RSS
with `canvas_storage=ram` and 1165 MiB with `canvas_storage=disk` (about 55% lower).
Disk storage also uses reclaimable OS file-cache memory outside process RSS. Decoded output
pixel hashes matched. This isolates canvas/codec work; it does not predict
the percentage saved during model sampling.
