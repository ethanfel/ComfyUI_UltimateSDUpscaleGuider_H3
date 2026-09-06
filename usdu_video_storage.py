"""Execution-scoped RGB canvas with one decoded frame resident per access."""

import operator
from pathlib import Path
import tempfile

from PIL import Image


class DiskFrameCanvas:
    """Mutable PIL frame sequence backed by raw RGB files, not a PIL clip in RAM.

    Returned images own their pixels and remain valid after the canvas closes.
    No tensor/mmap views or image cache survive an indexed read.
    """

    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="usdu_canvas_", dir=directory)
        self.directory = Path(self._temporary.name)
        self.sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._temporary.cleanup()

    def __len__(self):
        return len(self.sizes)

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Canvas frame index out of range")
        return index

    def _path(self, index):
        return self.directory / f"{index:08d}.rgb"

    def frame_size(self, index):
        return self.sizes[self._index(index)]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = self._index(index)
        return Image.frombytes("RGB", self.sizes[index], self._path(index).read_bytes())

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def _write(self, index, image):
        if image.mode != "RGB":
            raise ValueError("USDU's VIDEO canvas requires RGB frames.")
        self._path(index).write_bytes(image.tobytes())

    def __setitem__(self, index, image):
        index = self._index(index)
        self._write(index, image)
        self.sizes[index] = image.size

    def append(self, image):
        self._write(len(self), image)
        self.sizes.append(image.size)
