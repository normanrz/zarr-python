from __future__ import annotations

from typing import (
    Literal,
    Optional,
)

from dataclasses import dataclass
from numcodecs.gzip import GZip

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import JSON, BytesLike, to_thread
from zarr.v3.metadata import ChunkMetadata


@dataclass(frozen=True)
class GzipCodec(BytesBytesCodec):
    level: int = 5

    name: Literal["gzip"] = "gzip"
    is_fixed_size = True

    def to_json(self) -> JSON:
        return {**super().to_json(), "configuration": {"level": self.level}}

    async def decode(self, chunk_bytes: bytes, _chunk_metadata: ChunkMetadata) -> BytesLike:
        return await to_thread(GZip(self.level).decode, chunk_bytes)

    async def encode(
        self, chunk_bytes: bytes, _chunk_metadata: ChunkMetadata
    ) -> Optional[BytesLike]:
        return await to_thread(GZip(self.level).encode, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("gzip", GzipCodec)
