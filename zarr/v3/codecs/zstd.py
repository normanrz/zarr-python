from __future__ import annotations

from typing import (
    Literal,
    Optional,
)

from dataclasses import dataclass
from zstandard import ZstdCompressor, ZstdDecompressor

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import JSON, BytesLike, to_thread
from zarr.v3.metadata import ChunkMetadata


@dataclass(frozen=True)
class ZstdCodec(BytesBytesCodec):
    level: int = 0
    checksum: bool = False

    name: Literal["zstd"] = "zstd"
    is_fixed_size = True

    def to_json(self) -> JSON:
        return {
            **super().to_json(),
            "configuration": {"level": self.level, "checksum": self.checksum},
        }

    def _compress(self, data: bytes) -> bytes:
        ctx = ZstdCompressor(level=self.level, write_checksum=self.checksum)
        return ctx.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        ctx = ZstdDecompressor()
        return ctx.decompress(data)

    async def decode(self, chunk_bytes: bytes, _chunk_metadata: ChunkMetadata) -> BytesLike:
        return await to_thread(self._decompress, chunk_bytes)

    async def encode(
        self, chunk_bytes: bytes, _chunk_metadata: ChunkMetadata
    ) -> Optional[BytesLike]:
        return await to_thread(self._compress, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("zstd", ZstdCodec)
