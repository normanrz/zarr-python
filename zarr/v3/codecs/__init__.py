from __future__ import annotations
from dataclasses import dataclass

from functools import reduce
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

import numpy as np

from zarr.v3.abc.codec import Codec, ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.v3.common import BytesLike
from zarr.v3.metadata import ShardingCodecIndexLocation

if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata
    from zarr.v3.codecs.sharding import ShardingCodec
    from zarr.v3.codecs.blosc import BloscCodec
    from zarr.v3.codecs.bytes import BytesCodec
    from zarr.v3.codecs.transpose import TransposeCodec
    from zarr.v3.codecs.gzip import GzipCodec
    from zarr.v3.codecs.zstd import ZstdCodec
    from zarr.v3.codecs.crc32c_ import Crc32cCodec


@dataclass(frozen=True)
class CodecPipeline:
    codecs: List[Codec]

    @classmethod
    def create(
        cls,
        codecs: Iterable[Codec],
    ) -> CodecPipeline:
        out: List[Codec] = []
        for codec in codecs or []:
            # codec = codec.validate_evolve(chunk_metadata)
            out.append(codec)
            # chunk_metadata = codec.resolve_metadata(chunk_metadata)
        # CodecPipeline._validate_codecs(out, chunk_metadata)
        return cls(out)

    @staticmethod
    def _validate_codecs(codecs: List[Codec], chunk_metadata: ChunkMetadata) -> None:
        from zarr.v3.codecs.sharding import ShardingCodec

        assert any(
            isinstance(codec, ArrayBytesCodec) for codec in codecs
        ), "Exactly one array-to-bytes codec is required."

        prev_codec: Optional[Codec] = None
        for codec in codecs:
            if prev_codec is not None:
                assert not isinstance(codec, ArrayBytesCodec) or not isinstance(
                    prev_codec, ArrayBytesCodec
                ), (
                    f"ArrayBytesCodec '{type(codec)}' cannot follow after "
                    + f"ArrayBytesCodec '{type(prev_codec)}' because exactly "
                    + "1 ArrayBytesCodec is allowed."
                )
                assert not isinstance(codec, ArrayBytesCodec) or not isinstance(
                    prev_codec, BytesBytesCodec
                ), (
                    f"ArrayBytesCodec '{type(codec)}' cannot follow after "
                    + f"BytesBytesCodec '{type(prev_codec)}'."
                )
                assert not isinstance(codec, ArrayArrayCodec) or not isinstance(
                    prev_codec, ArrayBytesCodec
                ), (
                    f"ArrayArrayCodec '{type(codec)}' cannot follow after "
                    + f"ArrayBytesCodec '{type(prev_codec)}'."
                )
                assert not isinstance(codec, ArrayArrayCodec) or not isinstance(
                    prev_codec, BytesBytesCodec
                ), (
                    f"ArrayArrayCodec '{type(codec)}' cannot follow after "
                    + f"BytesBytesCodec '{type(prev_codec)}'."
                )

            if isinstance(codec, ShardingCodec):
                assert len(codec.chunk_shape) == len(chunk_metadata.shape), (
                    "The shard's `chunk_shape` and array's `shape` need to have the "
                    + "same number of dimensions."
                )
                assert all(
                    s % c == 0
                    for s, c in zip(
                        chunk_metadata.chunk_shape,
                        codec.chunk_shape,
                    )
                ), (
                    "The array's `chunk_shape` needs to be divisible by the "
                    + "shard's inner `chunk_shape`."
                )
            prev_codec = codec

        if any(isinstance(codec, ShardingCodec) for codec in codecs) and len(codecs) > 1:
            warn(
                "Combining a `sharding_indexed` codec disables partial reads and "
                + "writes, which may lead to inefficient performance."
            )

    def _array_array_codecs(self) -> List[ArrayArrayCodec]:
        return [codec for codec in self.codecs if isinstance(codec, ArrayArrayCodec)]

    def _array_bytes_codec(self) -> ArrayBytesCodec:
        return next(codec for codec in self.codecs if isinstance(codec, ArrayBytesCodec))

    def _bytes_bytes_codecs(self) -> List[BytesBytesCodec]:
        return [codec for codec in self.codecs if isinstance(codec, BytesBytesCodec)]

    async def decode(self, chunk_bytes: BytesLike, chunk_metadata: ChunkMetadata) -> np.ndarray:
        # TODO: resolve metadata in reverse order
        for bb_codec in self._bytes_bytes_codecs()[::-1]:
            chunk_bytes = await bb_codec.decode(chunk_bytes, chunk_metadata)

        chunk_array = await self._array_bytes_codec().decode(chunk_bytes, chunk_metadata)

        for aa_codec in self._array_array_codecs()[::-1]:
            chunk_array = await aa_codec.decode(chunk_array, chunk_metadata)

        return chunk_array

    async def encode(
        self, chunk_array: np.ndarray, chunk_metadata: ChunkMetadata
    ) -> Optional[BytesLike]:
        for aa_codec in self._array_array_codecs():
            chunk_array_maybe = await aa_codec.encode(chunk_array, chunk_metadata)
            chunk_metadata = aa_codec.resolve_metadata(chunk_metadata)
            if chunk_array_maybe is None:
                return None
            chunk_array = chunk_array_maybe

        ab_codec = self._array_bytes_codec()
        chunk_bytes_maybe = await ab_codec.encode(chunk_array, chunk_metadata)
        chunk_metadata = ab_codec.resolve_metadata(chunk_metadata)
        if chunk_bytes_maybe is None:
            return None
        chunk_bytes = chunk_bytes_maybe

        for bb_codec in self._bytes_bytes_codecs():
            chunk_bytes_maybe = await bb_codec.encode(chunk_bytes, chunk_metadata)
            chunk_metadata = bb_codec.resolve_metadata(chunk_metadata)
            if chunk_bytes_maybe is None:
                return None
            chunk_bytes = chunk_bytes_maybe

        return chunk_bytes

    def compute_encoded_size(self, byte_length: int) -> int:
        return reduce(lambda acc, codec: codec.compute_encoded_size(acc), self.codecs, byte_length)


def blosc_codec(
    typesize: int,
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd",
    clevel: int = 5,
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "noshuffle",
    blocksize: int = 0,
) -> "BloscCodec":
    from zarr.v3.codecs.blosc import BloscCodec

    return BloscCodec(
        cname=cname,
        clevel=clevel,
        shuffle=shuffle,
        blocksize=blocksize,
        typesize=typesize,
    )


def bytes_codec(endian: Optional[Literal["big", "little"]] = "little") -> "BytesCodec":
    from zarr.v3.codecs.bytes import BytesCodec

    return BytesCodec(endian=endian)


def transpose_codec(
    order: Union[Tuple[int, ...], Literal["C", "F"]], ndim: Optional[int] = None
) -> "TransposeCodec":
    from zarr.v3.codecs.transpose import TransposeCodec

    if order == "C" or order == "F":
        assert (
            isinstance(ndim, int) and ndim > 0
        ), 'When using "C" or "F" the `ndim` argument needs to be provided.'
        if order == "C":
            order = tuple(range(ndim))
        if order == "F":
            order = tuple(ndim - i - 1 for i in range(ndim))

    return TransposeCodec(order=order)


def gzip_codec(level: int = 5) -> "GzipCodec":
    from zarr.v3.codecs.gzip import GzipCodec

    return GzipCodec(level=level)


def zstd_codec(level: int = 0, checksum: bool = False) -> "ZstdCodec":
    from zarr.v3.codecs.zstd import ZstdCodec

    return ZstdCodec(level=level, checksum=checksum)


def crc32c_codec() -> "Crc32cCodec":
    from zarr.v3.codecs.crc32c_ import Crc32cCodec

    return Crc32cCodec()


def sharding_codec(
    chunk_shape: Tuple[int, ...],
    codecs: Optional[List[Codec]] = None,
    index_codecs: Optional[List[Codec]] = None,
    index_location: ShardingCodecIndexLocation = ShardingCodecIndexLocation.end,
) -> "ShardingCodec":
    from zarr.v3.codecs.sharding import ShardingCodec

    codecs = codecs or [bytes_codec()]
    index_codecs = index_codecs or [bytes_codec(), crc32c_codec()]
    return ShardingCodec(
        chunk_shape=chunk_shape,
        codecs=codecs,
        index_codecs=index_codecs,
        index_location=index_location,
    )
