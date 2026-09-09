Research snapshot from September 9, 2026 against node commit `5ea96ab` and local ComfyUI checkout `15eb748b`. This describes the baseline before implementation; see README.md for the subsequently implemented controls and compatibility behavior.

**Recommendation**

Improve H3 correctness and context anchoring first, then optimize tile storage and evaluate a separate latent refinement engine. The existing native VIDEO transport and disk canvas are useful foundations. Their memory savings do not address the full-clip tensor, H3 sampling memory, or differences between independently sampled tiles.

This review combines local source inspection, CPU diagnostics, current upstream code and issue history, and related implementations and papers. Proposed visual improvements remain hypotheses until tested with real H3 renders. No production code or workflow was changed.

**Highest-priority findings**

| Priority | Finding | Recommended change | Evidence / effort |
|---|---|---|---|
| 1 | H3 mask behavior depends on a recent core correction | Verify a core build containing PR #15988 before enabling H3 anchoring | Upstream fix merged September 8; absent from inspected local source. Small dependency task, render validation required |
| 1 | H3 creates an audio-lock mask but drops it; `anchor_context` is skipped | Implement explicit, tested nested video/audio masks and truthful compatibility behavior | Reproduced at the sampler boundary. Medium effort |
| 1 | H3 tile batching bypasses AV latent preparation | Reject `batch_size > 1` for H3 until true independent-clip batching exists | Reproduced. Small guard; large batching implementation |
| 1 | Allowed padding can create invalid H3 tile dimensions | Make the tile planner aware of the model's 32-pixel canvas alignment | Reproduced with 64-pixel tiles and 16-pixel padding. Small to medium effort |
| 1 | `seam_fix_denoise` does not change the Guider sampling schedule | Add explicit seam SIGMAS, preserving the main schedule | Reproduced with 0.1 versus 0.9. Small to medium effort |
| 2 | Disk canvas reads and writes full frames for each tile | Add region reads/writes and composite only the affected rectangle | Confirmed from source; performance gain unmeasured. Medium effort |
| 2 | Each tile is encoded, denoised and decoded independently | Prototype shared noise and synchronized latent refinement | Supported by related methods; H3 result unverified. Large effort |
| 3 | Each spatial tile contains the entire clip | Add optional temporal windows with context and overlap | Large potential memory benefit; temporal artifacts need evaluation. Large effort |

**1. Make H3 masks correct before turning them on**

In `modules/processing.py:604`, the helper constructs a nested video/audio latent and a mask with editable video and locked audio. At line 616 it returns only `samples`. `sample_with_guider` obtains its mask from `latent.get("noise_mask")`, so it receives none. The log still says audio is locked.

Separately, line 774 explicitly excludes H3 from `anchor_context`. A region mask still limits pixel compositing and can skip tiles; context-only overlap still changes what is composited. Neither mechanism currently freezes H3's context during denoising.

The distinction matters: a tile can generate against changing neighboring content even though those changes are discarded when the tile is pasted back. Restoring a stable boundary is a promising seam improvement, but it needs the correct H3 mask semantics.

Core recently had a mask regression. [ComfyUI PR #15988](https://github.com/Comfy-Org/ComfyUI/pull/15988), merged September 8, corrects velocity conversion for masked H3 video/audio. The inspected local `comfy/ldm/minimax/model.py` lacks those added multiplications. This is a finding about the checkout, not proof of the version loaded in a running server. The relevant [artifact report](https://github.com/Comfy-Org/ComfyUI/issues/15981) is closed through that fix.

The fork's oldest locally available commit, `6836cf3`, also has the message “Audio mask issue fixing.” That makes a past workaround plausible, but the shallow history does not establish why the mask was omitted. Do not blindly restore it and call the job complete.

Proposed implementation: build a nested mask containing a video edit region and a deliberate audio policy; align it with H3's spatial and temporal representation; verify it reaches the real sampler. Keep soft pixel feathering separate from the model's mask representation. Compare no mask, audio-only locking, and spatial anchoring on the same clip and seed.

The native VIDEO output already copies source audio from the input file. Locking the model's zero-filled audio latent is a different operation; it neither supplies the source soundtrack as conditioning nor establishes a speedup. Its effect on video quality must be measured.

**2. Prevent unsupported H3 execution paths**

`usdu_patch.py:272` combines frames and spatial tiles into one image batch. Its generic VAE path at line 282 bypasses `_usdu_h3_startlatent_prepare`. With H3 this mixes tile identity into the temporal input and fails to construct the nested AV pair.

A CPU diagnostic using five frames and two tiles reached sampling with one ordinary video tensor instead of a nested video/audio latent. The diagnostic intentionally stopped there; it did not attempt an actual model render.

Reject H3 `batch_size > 1` before decoding the source. Rename or clarify the control as spatial tile batching. A future implementation must represent independent tile clips separately and validate model batching support; concatenating them into time is unsuitable.

The current general planner also rounds to multiples of eight. A permitted 64-pixel tile plus 16-pixel padding becomes 80×80 and fails the H3 helper's 32-pixel check. Plan valid processing dimensions before allocating tensors, pad without distorting the crop, and crop back to preserve output size. Apply this to edge tiles and all seam modes.

Native H3 distinguishes a 32-pixel canvas grid from its 16× spatial VAE compression and uses a `17k+5` frame grid with nonuniform frame-to-token mapping. These are separate constraints. Reuse tested geometry helpers through a small compatibility layer. [Core H3 implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py).

**3. Give the seam pass a real sampling schedule**

The A1111 seam implementation writes `p.denoising_strength`; the Guider path always samples with `p.sigmas`. Changing `seam_fix_denoise` therefore does not change its actual noise schedule. Diagnostics with Half Tile mode and strengths 0.1 and 0.9 passed identical SIGMAS to all three sampler calls.

Add an optional `seam_sigmas` input and pass it only during seam processing. Preserve existing graphs when it is absent, and explain the legacy scalar's behavior. A distinct schedule is clearer than inventing a mapping from a scalar into an arbitrary caller-supplied schedule.

**4. Improve context and noise across tiles**

The Guider path passes the original guider directly to every tile. Unlike the non-Guider path, it does not adapt conditioning to the crop. Global reference material can remain global, but spatially aligned keyframes and guides require deliberate transformation into the tile's coordinates. Crop/resize those guides with the same geometry as the source tile, preserve reference labels and timing, and isolate per-tile conditioning without mutating the caller's guider. Validate against text-only, keyframe and reference-conditioned workflows.

Every tile also restarts noise generation with the same seed. For equal-shaped tiles this repeats local noise; it does not give the same global overlap location matching noise when that location has a different local coordinate.

An intermediate experiment is a reproducible global noise field from which tiles take coordinate-aligned crops. Avoid allocating an enormous full-video noise tensor merely to obtain consistency: deterministic blocks or CPU/disk storage are possible implementations. Padding, resizing, and additional stochastic sampler noise must follow the same coordinates. This requires an explicit geometry contract before it is useful.

The larger experiment is synchronized latent refinement: hold a shared latent, obtain predictions from overlapping tiles at a sampling step, combine them, and advance a consistent sampler state. Weighted prediction fusion is grounded in [MultiDiffusion](https://arxiv.org/abs/2302.08113). Its results do not establish H3 video quality, and support for multistep/stochastic samplers needs explicit design.

The upstream successor is now [Context-Anchored Tile Refine](https://github.com/Blakeem/ComfyUI-ContextAnchoredTileRefine). Its base node uses frozen neighbor context; its VL nodes use synchronized latent tiling. The documented VL path targets Krea 2, so it is a design reference rather than demonstrated H3 compatibility. The old successor URL in this fork returned 404 during this review.

Preserve the current pixel engine as a baseline while prototyping this separately. Encoding the source once and decoding a consolidated result can reduce repeated VAE work and quantization, but changes how neighboring refinements feed later tiles.

**5. Reduce the next memory and I/O bottlenecks**

`DiskFrameCanvas.__getitem__` reads an entire RGB frame. `CroppedImages` then extracts its tile. Compositing reads the full frame again, allocates multiple full-frame RGBA images, and writes the entire RGB frame back.

Add `read_region` and `write_region` operations, with bounded row reads or carefully scoped mmap access to raw storage. Composite a rectangle using the corresponding mask, retaining the existing alpha behavior. Keep RAM and disk implementations under the same interface and require pixel equality before treating this as an optimization.

For scale, 175 frames of 3840×2160 RGB8 occupy about 4.06 GiB. Reading the full clip twice and writing it once per spatial tile implies roughly 12.17 GiB of logical canvas traffic per tile, excluding output transport. Filesystem caching means this is not a prediction of physical SSD traffic. This should be measured with actual refinement enabled.

There is still a full temporal float buffer in `pil_batch_to_tensor`: 175 frames at 512×512 RGB float32 require 525 MiB, before VAE outputs and model memory. Disk mode does not remove it. Native H3 VAE temporal processing can transfer smaller clips internally, but the caller has already materialized the tile tensor.

Expose timings for source decode, crop/convert, VAE encode, sampling, VAE decode, composite and output encode. Report peak process RSS, GPU allocated/reserved memory and scratch usage separately. Prefer measured bottlenecks over a generic “low VRAM” preset.

**6. Add temporal windows as an optional mode**

Process shorter temporal windows with overlapping context; carry an anchored region from the previous window; coordinate noise across overlap; retain exact source timestamps and audio; reset at shot changes. Window starts, lengths and guide timestamps need H3-aware mapping. Do not use a generic constant frames-per-token conversion.

[MMH3 Split Upscale](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) documents temporal/spatial splitting, overlap and anchors on H3 AV latents. It is a useful implementation to compare, but its seam and quality claims were not independently reproduced here. [Upscale-A-Video](https://arxiv.org/abs/2312.06640) provides research support for propagating temporal information across sequences; transferring its method to H3 would be new engineering work.

Shorter windows can lower working memory but may lose motion context and create joins. Keep whole-clip processing available for short clips and for quality comparisons. First stabilize spatial refinement; otherwise temporal and spatial boundaries become difficult to diagnose separately.

**Additional improvements**

| Area | Concrete action |
|---|---|
| Precision | `tensor_to_pil` quantizes to uint8 after `nan_to_num` without clamping. Clamp before conversion to prevent out-of-range wraparound. Evaluate a separate float16/uint16 canvas; RGB16 output alone cannot recover lost precision. |
| Color | Record transfer/primaries/range metadata and define supported input handling. Test gradients and shadows. Treat per-frame color matching cautiously because it can introduce temporal pumping. |
| FPS and trims | Define behavior for non-24-fps/VFR input: preserving file timestamps does not adapt H3's internal motion clock. Add bounded streaming trim/crop support only with corresponding timestamp and audio handling. |
| Validation | Check model/VAE compatibility, effective tile dimensions, clip length, mask length, SIGMAS and scratch capacity before loading the full scene. Current H3 class-name detection can miss subclasses. |
| Observability | Replace global root-logger suppression with selective logging. Show effective processing dimensions, tile count, sampling steps and elapsed stage times. |
| Packaging | Fix inherited deprecation/publisher/repository metadata and stale links. Replace import-time downloading of an unpinned upstream ZIP with a reproducible dependency/install path. |
| Maintenance | Extract an H3 adapter and run-scoped state. Keep the current cleanup guarantees while reducing dependence on A1111 monkey patches and global module manipulation. |

**Validation performed and proposed**

All existing lightweight suites passed with Python from `/media/p5/miniforge3/envs/13_env_py313` and `COMFYUI_ROOT=/media/p5/Comfyui`: five memory tests, seven VIDEO tests, and three disk-canvas tests, 15 total. These use CPU transport checks and sampling/VAE stand-ins; they do not demonstrate real H3 quality or inference performance.

Additional CPU diagnostics exercised the real node/tile loops with deterministic VAE/sampler stand-ins:

| Case | Observed result |
|---|---|
| Baseline H3 | Nested AV samples reached sampling with no `noise_mask` |
| H3 `anchor_context=True` | Still no `noise_mask` |
| H3 `batch_size=2` | Ordinary video tensor reached sampling; AV pair missing |
| Half Tile seam strength 0.1 vs 0.9 | Identical SIGMAS supplied to each sampler call |
| Tile 64, padding 16 | Rejected as invalid 80×80 H3 geometry before sampling |

For implementation, retain those cases as meaningful regressions and add a small fixed H3 render set: a static face with fine texture; a subject crossing a spatial boundary; a camera pan over repeating lines; a dark gradient; and a clip with a shot change. Include frame counts around the valid grid, such as 5, 21, 22, 23, 39 and 175, plus odd output dimensions and per-frame masks.

Compare one change at a time using the same source, prompt, weights, sampler, SIGMAS and seed. Measure visible seam strength, motion-compensated temporal error with occlusion handling, detail/identity retention, color drift, stage time, RSS and VRAM. Inspect motion directly: a lower temporal error can also mean unwanted smoothing. Verify output frame count, timestamps and audio independently of visual metrics.

Suggested implementation order: geometry/batching guards and truthful controls; core compatibility and H3 masks; seam SIGMAS; profiling and region canvas I/O; then separate shared-latent and temporal-window experiments. No speedup percentage or visual-quality gain is claimed without those measurements.
