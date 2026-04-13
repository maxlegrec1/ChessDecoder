"""Minimal reader for DeepMind searchless_chess `.bag` files.

Vendored down from https://github.com/google-deepmind/searchless_chess/blob/main/src/bagz.py
to avoid the heavy `apache_beam` / `etils` / `pygrain` deps. Supports only
uncompressed `.bag` files (no `.bagz` zstd). Exposes:

    * BagFileReader(path)            — random-access sequence of raw bytes records
    * iter_action_value_records(path) → Iterator[(fen, move, win_prob)]

The `.bag` layout (from searchless_chess/src/bagz.py):

    [record_0_bytes][record_1_bytes]...[record_N-1_bytes][limit_table]

with `limit_table` being N int64-little-endian end-offsets (cumulative byte
positions inside the records region). The very last 8 bytes of the file double
as the start-of-limits pointer.

The action-value record schema (from searchless_chess/src/constants.py) is the
Apache Beam TupleCoder of (StrUtf8Coder, StrUtf8Coder, FloatCoder), which on
the wire is::

    varint(len(fen)) | fen_utf8 | varint(len(move)) | move_utf8 | be_double(win_prob)

(Beam's Python `FloatCoder` is actually 8-byte big-endian double.)
"""

from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Iterator, Sequence
from typing import SupportsIndex


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Beam-style varint: 7 bits per byte, high bit = continuation."""
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def decode_action_value(rec: bytes) -> tuple[str, str, float]:
    """Decode one (fen, move, win_prob) action-value record."""
    fen_len, pos = _read_varint(rec, 0)
    fen = rec[pos:pos + fen_len].decode("utf-8")
    pos += fen_len
    move_len, pos = _read_varint(rec, pos)
    move = rec[pos:pos + move_len].decode("utf-8")
    pos += move_len
    (win_prob,) = struct.unpack(">d", rec[pos:pos + 8])
    return fen, move, win_prob


class BagFileReader(Sequence[bytes]):
    """Random-access reader for a single uncompressed `.bag` file."""

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = os.fspath(path)
        size = os.path.getsize(self._path)
        if size < 8:
            raise ValueError(f"{self._path}: file too small ({size} bytes)")

        fd = os.open(self._path, os.O_RDONLY)
        try:
            self._mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)

        (index_start,) = struct.unpack("<Q", self._mm[-8:])
        index_size = size - index_start
        if index_size < 0 or index_size % 8 != 0:
            raise ValueError(
                f"{self._path}: bad index size {index_size} "
                f"(file_size={size}, index_start={index_start})"
            )
        self._num_records = index_size // 8
        self._index_start = index_start
        # Decode the entire limit table once — it's small (8 bytes per record).
        self._limits = struct.unpack(
            f"<{self._num_records}q",
            self._mm[index_start:size],
        )

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, index: SupportsIndex) -> bytes:  # type: ignore[override]
        i = index.__index__()
        if i < 0:
            i += self._num_records
        if not 0 <= i < self._num_records:
            raise IndexError(i)
        end = self._limits[i]
        start = 0 if i == 0 else self._limits[i - 1]
        return self._mm[start:end]


def iter_action_value_records(
    path: str | os.PathLike,
) -> Iterator[tuple[str, str, float]]:
    """Stream (fen, move, win_prob) tuples from an action-value `.bag` file."""
    reader = BagFileReader(path)
    for i in range(len(reader)):
        yield decode_action_value(reader[i])
