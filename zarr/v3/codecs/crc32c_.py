from __future__ import annotations

from typing import (
    Literal,
    Optional,
)

import numpy as np
from dataclasses import dataclass
from crc32c import crc32c

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import BytesLike
from zarr.v3.metadata import ChunkMetadata


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    name: Literal["crc32c"] = "crc32c"
    is_fixed_size = True

    async def decode(self, chunk_bytes: bytes, _chunk_metadata: ChunkMetadata) -> BytesLike:
        crc32_bytes = chunk_bytes[-4:]
        inner_bytes = chunk_bytes[:-4]

        assert np.uint32(crc32c(inner_bytes)).tobytes() == bytes(crc32_bytes)
        return inner_bytes

    async def encode(
        self, chunk_bytes: bytes, _chunk_metadata: ChunkMetadata
    ) -> Optional[BytesLike]:
        return chunk_bytes + np.uint32(crc32c(chunk_bytes)).tobytes()

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length + 4


register_codec("crc32c", Crc32cCodec)
