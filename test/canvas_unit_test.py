"""CPU pixel/I/O regressions: python -B test/canvas_unit_test.py."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from usdu_canvas import RGB16Frame, composite_tile, frame_from_pixels, frame_pixels, frame_reference
from usdu_video_storage import DiskFrameCanvas


def legacy_composite(base, tile, region, mask):
    layer = Image.new("RGBA", base.size)
    layer.paste(tile, region[:2])
    temp = layer.copy()
    temp.putalpha(mask)
    layer.paste(temp, layer)
    result = base.convert("RGBA")
    result.alpha_composite(layer)
    return result.convert("RGB")


class CanvasTests(unittest.TestCase):
    def test_region_blend_matches_legacy_uint8_exactly(self):
        rng = np.random.default_rng(81)
        for box in ((7, 9, 27, 32), (-4, -7, 24, 33), (20, 25, 49, 51), (0, 0, 43, 39)):
            pixels = rng.integers(0, 256, (39, 43, 3), dtype=np.uint8)
            tile = Image.fromarray(rng.integers(0, 256, (box[3] - box[1], box[2] - box[0], 3), dtype=np.uint8))
            # Exhaust the 256 alpha values as well as random source/dest colors.
            mask = Image.fromarray(np.resize(np.arange(256, dtype=np.uint8), (39, 43)))
            expected = legacy_composite(Image.fromarray(pixels), tile, box, mask)
            frames = [Image.fromarray(pixels)]
            composite_tile(frames, 0, tile, box, mask)
            self.assertEqual(frames[0].tobytes(), expected.tobytes())
            with tempfile.TemporaryDirectory() as directory, DiskFrameCanvas(directory) as disk:
                disk.append(Image.fromarray(pixels))
                composite_tile(disk, 0, tile, box, mask)
                self.assertEqual(disk[0].tobytes(), expected.tobytes())

    def test_uint16_resize_and_crop_keep_intermediate_levels(self):
        pixels = np.arange(1024, dtype=np.uint16)[None, :, None].repeat(3, axis=2)
        frame = RGB16Frame(pixels)
        enlarged = frame.resize((2048, 4))
        self.assertEqual(enlarged.pixels.dtype, np.uint16)
        self.assertGreater(len(np.unique(enlarged.pixels)), 1000)
        self.assertLessEqual(enlarged.pixels.max(), 1024)
        padded = frame.crop((-2, -1, 1026, 2)).pixels
        np.testing.assert_array_equal(padded[1:2, 2:-2], pixels)
        self.assertFalse(padded[0].any())
        self.assertFalse(padded[:, :2].any())
        self.assertFalse(padded[:, -2:].any())

    def test_uint16_blend_preserves_masked_pixels_and_small_changes(self):
        pixels = np.full((8, 16, 3), 30000, dtype=np.uint16)
        tile = RGB16Frame(np.full((8, 16, 3), 30004, dtype=np.uint16))
        weights = np.array([0, 128, 255, 64], dtype=np.uint8).repeat(4)
        mask = Image.fromarray(np.tile(weights, (8, 1)))
        expected = np.tile(np.repeat([30000, 30002, 30004, 30001], 4)[None, :, None], (8, 1, 3))
        frames = [RGB16Frame(pixels.copy())]
        composite_tile(frames, 0, tile, (0, 0, 16, 8), mask)
        np.testing.assert_array_equal(frames[0].pixels, expected)
        with tempfile.TemporaryDirectory() as directory, DiskFrameCanvas(directory, "16-bit") as disk:
            disk.append(RGB16Frame(pixels))
            composite_tile(disk, 0, tile, (0, 0, 16, 8), mask)
            np.testing.assert_array_equal(disk[0].pixels, expected)

    def test_disk_region_access_counts_only_requested_bytes(self):
        original_open = Path.open
        counts = {"read": 0, "write": 0}

        class CountedFile:
            def __init__(self, path, *args, **kwargs):
                self.stream = original_open(path, *args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def seek(self, offset):
                return self.stream.seek(offset)

            def readinto(self, buffer):
                counts["read"] += len(buffer)
                return self.stream.readinto(buffer)

            def write(self, buffer):
                counts["write"] += len(buffer)
                return self.stream.write(buffer)

        for precision, dtype in (("8-bit", np.uint8), ("16-bit", np.uint16)):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as directory, DiskFrameCanvas(directory, precision) as disk:
                pixels = np.arange(128 * 96 * 3).reshape(96, 128, 3).astype(dtype)
                frame = frame_from_pixels(pixels)
                disk.append(frame)
                self.assertEqual(disk._path(0).stat().st_size, pixels.nbytes)
                box = (11, 7, 31, 22)
                counts.update(read=0, write=0)
                with patch.object(Path, "open", new=lambda path, *a, **kw: CountedFile(path, *a, **kw)), \
                        patch.object(DiskFrameCanvas, "__getitem__", side_effect=AssertionError("full frame read")), \
                        patch.object(DiskFrameCanvas, "_write", side_effect=AssertionError("full frame write")):
                    ref = frame_reference(disk, 0)
                    self.assertEqual(ref.size, (128, 96))
                    self.assertIs(ref.resize(ref.size), ref)
                    disk[0] = ref
                    self.assertEqual(counts, {"read": 0, "write": 0})
                    tile = ref.crop(box)
                    np.testing.assert_array_equal(frame_pixels(tile), pixels[7:22, 11:31])
                    composite_tile(disk, 0, tile, box, Image.new("L", ref.size, 255))
                region_bytes = 20 * 15 * 3 * np.dtype(dtype).itemsize
                self.assertEqual(counts, {"read": 2 * region_bytes, "write": region_bytes})
                np.testing.assert_array_equal(frame_pixels(disk[0]), pixels)

    def test_disk_outside_crops_invalid_writes_and_short_reads(self):
        for precision, dtype in (("8-bit", np.uint8), ("16-bit", np.uint16)):
            with tempfile.TemporaryDirectory() as directory, DiskFrameCanvas(directory, precision) as disk:
                frame = frame_from_pixels(np.full((11, 13, 3), 17, dtype=dtype))
                disk.append(frame)
                for region in ((-3, -2, 6, 7), (9, 7, 19, 15), (20, 20, 23, 23)):
                    np.testing.assert_array_equal(frame_pixels(disk.read_region(-1, region)), frame_pixels(frame.crop(region)))
                with self.assertRaises(ValueError):
                    disk.write_region(0, (-1, 0, 12, 11), frame)
                disk._path(0).write_bytes(b"short")
                with self.assertRaises(OSError):
                    disk.read_region(0, (0, 0, 13, 11))


if __name__ == "__main__":
    unittest.main()
