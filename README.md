# ComfyUI_UltimateSDUpscaleGuider_H3

> **A fork of [ComfyUI_UltimateSDUpscaleGuider](https://github.com/Blakeem/ComfyUI_UltimateSDUpscaleGuider) with MiniMax H3 model support.**

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for running the image-to-image diffusion process on large images in tiles. Tiling improves the detail commonly lost on upscaled images while keeping VRAM use low and the working size close to what the diffusion model was trained on.

## Fork Changes

1. **MiniMax H3 model support**: original USDU nodes do not support MiniMax H3 model and produces errors when you try to connect it. This is a **vibe-coded** fork to bring H3 model support and use its potential for upscaling.

Please see example workflow: [minimax_h3_usdu.json](https://github.com/lisitskyaa/ComfyUI_UltimateSDUpscaleGuider_H3/blob/main/example_workflows/minimax_h3_usdu.json)

## H3 refinement controls

H3 uses one spatial tile containing the whole clip. Set `batch_size=1`;
unsupported spatial tile batching is rejected before the video is decoded.
Processing tiles are padded with edge pixels to a multiple of 32 and cropped
back after decoding, so padding and edge tiles do not have to match H3's grid.
The H3 output canvas keeps its requested dimensions, including odd sizes.

- `anchor_context`: now works with H3 when a region mask is connected or
  `tile_overlap_mode` is `Context Only Overlap`. It holds the surrounding
  context during sampling. A model patch is editable if any of its pixels or
  represented video frames are editable; the pixel mask still controls the
  exact final boundary and feathering. Use spatial padding to provide context.
- `h3_audio_lock`: optionally holds H3's internal empty audio latent fixed.
  Default **off** preserves existing sampling behavior. This can change the
  video result; it does not feed the original soundtrack into the model.
  The VIDEO node copies source audio independently of this setting.
- `seam_sigmas`: an optional separate schedule for seam repair, available on
  every Guider node. Connect a second scheduler to control seam denoise and
  steps independently. Without this input, seams use the main `sigmas`.
  The legacy `seam_fix_denoise` widget remains for workflow compatibility;
  it does not control Guider sampling.

Masked H3 runs check the installed model's velocity conversion on small CPU
tensors without loading weights. For native builds missing
[ComfyUI's mask correction](https://github.com/Comfy-Org/ComfyUI/pull/15988),
the node applies the correction through a wrapper on its own copied guider.
Builds that already implement it receive no additional correction. This does
not edit ComfyUI files or attach the wrapper to the input guider. An
unrecognized conversion fails before source video decoding.

These changes have CPU regression coverage for geometry, masks, audio
conversion, schedules and cleanup. Their effect on visual quality still
needs comparison with real H3 renders.

## Canvas precision

All Guider IMAGE and VIDEO nodes have a `canvas_precision` switch:

- `8-bit` (default): preserves the existing 256 levels per RGB channel and
  the original rounding and blending results.
- `16-bit`: stores 65,536 levels per RGB channel at input and after each VAE
  decode. Resizing uses CPU float32 channels before rounding back to uint16;
  blending also retains 16-bit RGB precision. This reduces accumulated rounding
  in gradients and repeated tile/seam passes. It cannot recover detail already
  lost in an 8-bit source.

16-bit doubles the canvas pixel storage in RAM or on disk. Tile buffers sent to
the VAE remain float32 with the same shape, and model/latent precision is
unchanged, so this switch is not expected to increase sampling VRAM. CPU resize
and blend work can cost more. Masks retain their existing 8-bit precision.
IMAGE outputs remain ComfyUI float tensors; downstream save nodes determine
the saved bit depth. Native VIDEO output already supports RGB16.

## VIDEO refinement and memory

`Ultimate SD Upscale (No Upscale, Guider, VIDEO)` accepts a file-backed native
ComfyUI `VIDEO`, such as Load Video or the output of the H3/DLSS5 VIDEO pipeline.
It inherits the IMAGE node's controls and uses the same tile and sampling engine.
The CAT pack is not required. Update both packs to make older workflows using
`CATUltimateSDUpscaleGuiderVideo` forward to this node automatically.

The VIDEO path decodes and encodes one frame at a time, avoiding the previous
adapter's full float32 input mmap and output tensor. Its optional `canvas_storage`
switch selects where the editable RGB canvas lives:

- `ram` (default): keep RGB frames in memory.
- `disk`: keep raw RGB frame files in temporary storage. Each tile crop and
  blend reads/writes only its rectangle, including its padding. Full frames are
  transferred during input/output and when the canvas must be resized.
  Use fast local scratch storage for large clips under RAM pressure.
  Files require `frames × width × height × 3 × bytes_per_channel` bytes
  (`bytes_per_channel` is 1 for 8-bit or 2 for 16-bit) and are
  removed after success, failure or cancellation. OS file-cache pages are
  reclaimable but can still appear as used memory.

`canvas_directory` optionally chooses the scratch folder for Disk mode. Leave
it empty to use ComfyUI's temp folder, or set a folder on a local SSD. A
RAM-backed filesystem such as tmpfs still holds these files in RAM, so choose
real disk storage to reduce system RAM pressure. RAM mode ignores this field.

Both storage settings use the same pixels at the selected canvas precision,
tile order, sampling and temporal context.
The storage switch does not change denoiser VRAM requirements. Both IMAGE and
VIDEO paths crop and resize one tile frame at a time into the VAE buffer,
blend within tile regions, and release shared buffers on every exit.

Output is temporary FFV1 RGB16 Matroska with source timestamps and stream-copied
audio. There is no intermediate CRF or YUV subsampling. Select `canvas_precision=16-bit`
to preserve RGB16 precision inside the canvas as well as in transport. The default
8-bit canvas still quantizes the pixels before RGB16 output. Final delivery
encoding remains downstream. Lazy trimmed
or cropped VIDEO objects must first be saved and reloaded.

Tiling is spatial: `batch_size=1` means one tile containing the whole clip, not
one video frame. In RAM mode the RGB canvas still occupies memory; in both
modes the temporal float tile, model weights,
reference conditioning and sampling activations still require memory. This
change reduces RAM allocations; it does not introduce temporal chunking or
reduce the denoiser's VRAM requirement for the same settings. Smaller spatial
tiles/padding affect sampling memory; `tiled_decode` affects VAE decode memory.

CPU regressions (no model downloads or GPU rendering):

```sh
COMFYUI_ROOT=/path/to/ComfyUI python -B test/memory_unit_test.py
COMFYUI_ROOT=/path/to/ComfyUI python -B test/video_unit_test.py
COMFYUI_ROOT=/path/to/ComfyUI python -B test/h3_unit_test.py
COMFYUI_ROOT=/path/to/ComfyUI python -B test/precision_unit_test.py
python -B test/video_storage_unit_test.py
python -B test/canvas_unit_test.py
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
with `canvas_storage=ram` and 1165 MiB with `canvas_storage=disk` (about 55% lower),
using the original 8-bit canvas before region I/O was added.
Disk storage also uses reclaimable OS file-cache memory outside process RSS. Decoded output
pixel hashes matched. This isolates canvas/codec work; it does not predict
the percentage saved during model sampling.
