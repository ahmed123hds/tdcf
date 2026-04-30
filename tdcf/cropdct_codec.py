"""Compression codecs for CropDCT.

CropDCT uses independent compressed chunks for every spatial tile and
frequency band. This module keeps the compression backend isolated so the
physical layout does not depend on a Python package such as ``zstandard``.
"""

from __future__ import annotations

import ctypes
import ctypes.util


class ZstdCodec:
    """Small ctypes wrapper around libzstd.

    The implementation intentionally requires libzstd. If it is not available,
    building a CropDCT store should fail loudly instead of silently falling back
    to a different storage format.
    """

    name = "zstd"

    def __init__(self, level: int = 1):
        self.level = int(level)
        lib_path = ctypes.util.find_library("zstd")
        if lib_path is None:
            raise RuntimeError("libzstd is required for CropDCT compression")
        self._lib = ctypes.CDLL(lib_path)
        self._lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        self._lib.ZSTD_compressBound.restype = ctypes.c_size_t
        self._lib.ZSTD_compress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._lib.ZSTD_compress.restype = ctypes.c_size_t
        self._lib.ZSTD_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._lib.ZSTD_decompress.restype = ctypes.c_size_t
        self._lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self._lib.ZSTD_isError.restype = ctypes.c_uint
        self._lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        self._lib.ZSTD_getErrorName.restype = ctypes.c_char_p

    def _check(self, code: int):
        if self._lib.ZSTD_isError(code):
            msg = self._lib.ZSTD_getErrorName(code).decode("utf-8", errors="replace")
            raise RuntimeError(f"zstd error: {msg}")

    def compress(self, data: bytes) -> bytes:
        if not data:
            return b""
        src = (ctypes.c_char * len(data)).from_buffer_copy(data)
        bound = int(self._lib.ZSTD_compressBound(len(data)))
        dst = ctypes.create_string_buffer(bound)
        written = int(
            self._lib.ZSTD_compress(dst, bound, src, len(data), self.level)
        )
        self._check(written)
        return dst.raw[:written]

    def decompress(self, payload: bytes, raw_length: int) -> bytes:
        if raw_length == 0:
            return b""
        src = (ctypes.c_char * len(payload)).from_buffer_copy(payload)
        dst = ctypes.create_string_buffer(int(raw_length))
        written = int(
            self._lib.ZSTD_decompress(dst, int(raw_length), src, len(payload))
        )
        self._check(written)
        if written != int(raw_length):
            raise RuntimeError(
                f"zstd decompressed {written} bytes, expected {raw_length}"
            )
        return dst.raw


def make_codec(name: str, level: int = 1):
    if name != "zstd":
        raise ValueError(f"Unsupported CropDCT codec: {name!r}")
    return ZstdCodec(level=level)
