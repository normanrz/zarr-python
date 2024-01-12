from __future__ import annotations
from dataclasses import dataclass

from functools import reduce
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
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
from zarr.v3.metadata import ArrayMetadata, ShardingCodecIndexLocation

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

    def validate(self, array_metadata: ArrayMetadata) -> None:
        from zarr.v3.codecs.sharding import ShardingCodec

        assert any(
            isinstance(codec, ArrayBytesCodec) for codec in self.codecs
        ), "Exactly one array-to-bytes codec is required."

        prev_codec: Optional[Codec] = None
        for codec in self.codecs:
            codec.validate(array_metadata)
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
            prev_codec = codec

        if any(isinstance(codec, ShardingCodec) for codec in self.codecs) and len(self.codecs) > 1:
            warn(
                "Combining a `sharding_indexed` codec disables partial reads and "
                + "writes, which may lead to inefficient performance."
            )

    def _codecs_with_resolved_metadata(
        self, chunk_metadata: ChunkMetadata
    ) -> Iterator[Tuple[Codec, ChunkMetadata]]:
        for codec in self.codecs:
            yield (codec, chunk_metadata)
            chunk_metadata = codec.resolve_metadata(chunk_metadata)

    async def decode(self, chunk_bytes: BytesLike, chunk_metadata: ChunkMetadata) -> np.ndarray:
        codecs = list(self._codecs_with_resolved_metadata(chunk_metadata))[::-1]

        for bb_codec, chunk_metadata in filter(lambda c: isinstance(c[0], BytesBytesCodec), codecs):
            chunk_bytes = await bb_codec.decode(chunk_bytes, chunk_metadata)

        ab_codec, chunk_metadata = next(filter(lambda c: isinstance(c[0], ArrayBytesCodec), codecs))
        chunk_array = await ab_codec.decode(chunk_bytes, chunk_metadata)

        for aa_codec, chunk_metadata in filter(lambda c: isinstance(c[0], ArrayArrayCodec), codecs):
            chunk_array = await aa_codec.decode(chunk_array, chunk_metadata)

        return chunk_array

    async def encode(
        self, chunk_array: np.ndarray, chunk_metadata: ChunkMetadata
    ) -> Optional[BytesLike]:
        codecs = list(self._codecs_with_resolved_metadata(chunk_metadata))

        for aa_codec, chunk_metadata in filter(lambda c: isinstance(c[0], ArrayArrayCodec), codecs):
            chunk_array_maybe = await aa_codec.encode(chunk_array, chunk_metadata)
            if chunk_array_maybe is None:
                return None
            chunk_array = chunk_array_maybe

        ab_codec, chunk_metadata = next(filter(lambda c: isinstance(c[0], ArrayBytesCodec), codecs))
        chunk_bytes_maybe = await ab_codec.encode(chunk_array, chunk_metadata)
        if chunk_bytes_maybe is None:
            return None
        chunk_bytes = chunk_bytes_maybe

        for bb_codec, chunk_metadata in filter(lambda c: isinstance(c[0], BytesBytesCodec), codecs):
            chunk_bytes_maybe = await bb_codec.encode(chunk_bytes, chunk_metadata)
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
