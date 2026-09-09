"""CPU RGB frames and region compositing, with explicit 8/16-bit precision."""

import numpy as np
from PIL import Image


def validate_precision(precision):
    if precision not in ("8-bit", "16-bit"):
        raise ValueError("canvas_precision must be 8-bit or 16-bit.")
    return precision


class RGB16Frame:
    """uint16 RGB storage with the crop/resize surface used by the tile planner.

    Pillow has no RGB uint16 mode. Resample each channel in Pillow's float32
    mode on the CPU, then round back to uint16; never pass through RGB uint8.
    """

    mode = "RGB"

    def __init__(self, pixels):
        if pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("RGB16Frame requires an HWC uint16 RGB array.")
        self.pixels = np.ascontiguousarray(pixels)

    @property
    def size(self):
        return self.width, self.height

    @property
    def width(self):
        return self.pixels.shape[1]

    @property
    def height(self):
        return self.pixels.shape[0]

    def crop(self, region):
        x1, y1, x2, y2 = region
        if x2 < x1 or y2 < y1:
            raise ValueError("Invalid crop region.")
        result = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint16)
        left, top, right, bottom = clipped_region(region, self.size)
        if right > left and bottom > top:
            result[top - y1:bottom - y1, left - x1:right - x1] = self.pixels[top:bottom, left:right]
        return RGB16Frame(result)

    def resize(self, size, resample=Image.Resampling.LANCZOS):
        if tuple(size) == self.size:
            return self
        result = np.empty((size[1], size[0], 3), dtype=np.uint16)
        for channel in range(3):
            plane = Image.fromarray(self.pixels[:, :, channel].astype(np.float32))
            resized = np.asarray(plane.resize(size, resample=resample))
            result[:, :, channel] = np.rint(resized.clip(0, 65535)).astype(np.uint16)
        return RGB16Frame(result)

    def tobytes(self):
        return self.pixels.astype("<u2", copy=False).tobytes()


def frame_pixels(frame):
    return frame.pixels if isinstance(frame, RGB16Frame) else np.asarray(frame)


def frame_from_pixels(pixels):
    return RGB16Frame(pixels) if pixels.dtype == np.uint16 else Image.fromarray(pixels)


def clipped_region(region, size):
    x1, y1, x2, y2 = region
    return max(0, x1), max(0, y1), min(size[0], x2), min(size[1], y2)


class CanvasFrameReference:
    """Planner handle: dimensions cost no disk read; crop reads only its region."""

    mode = "RGB"

    def __init__(self, canvas, index):
        self.canvas = canvas
        self.index = index

    @property
    def size(self):
        return self.canvas.frame_size(self.index)

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def crop(self, region):
        return self.canvas.read_region(self.index, region)

    def resize(self, size, resample=Image.Resampling.LANCZOS):
        if tuple(size) == self.size:
            return self
        return self.canvas[self.index].resize(size, resample=resample)


def frame_reference(frames, index):
    if hasattr(frames, "read_region"):
        return CanvasFrameReference(frames, index)
    return frames[index]


def composite_tile(frames, index, tile, region, mask):
    """Read/blend/write only the tile rectangle. Preserve legacy uint8 rounding."""
    frame = frame_reference(frames, index)
    left, top, right, bottom = clipped_region(region, frame.size)
    if right <= left or bottom <= top:
        return
    box = (left, top, right, bottom)
    alpha = mask.crop(box)
    if alpha.getbbox() is None:
        return
    tile = tile.crop((left - region[0], top - region[1],
                      right - region[0], bottom - region[1]))
    base = frame.crop(box)
    if isinstance(base, RGB16Frame):
        if not isinstance(tile, RGB16Frame):
            raise ValueError("Cannot composite an 8-bit tile into a 16-bit canvas.")
        weight = np.asarray(alpha, dtype=np.uint32)[:, :, None]
        # Values remain below 65535 * 255; uint32 is sufficient and exact.
        pixels = (base.pixels.astype(np.uint32) * (255 - weight)
                  + tile.pixels.astype(np.uint32) * weight + 127) // 255
        result = RGB16Frame(pixels.astype(np.uint16))
    else:
        layer = tile.convert("RGBA")
        layer.putalpha(alpha)
        result = base.convert("RGBA")
        result.alpha_composite(layer)
        result = result.convert("RGB")
    if hasattr(frames, "write_region"):
        frames.write_region(index, box, result)
    elif isinstance(frame, RGB16Frame):
        frame.pixels[top:bottom, left:right] = result.pixels
    else:
        frame.paste(result, (left, top))
