from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
)

import numpy as np
from dataclasses import dataclass

from zarr.v3.abc.codec import ArrayBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import JSON, BytesLike

if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata, ArrayMetadata


Endian = Literal["big", "little"]


@dataclass(frozen=True)
class BytesCodec(ArrayBytesCodec):
    endian: Optional[Endian] = "little"

    name: Literal["bytes"] = "bytes"
    is_fixed_size = True

    def to_json(self) -> JSON:
        configuration_json = {}
        if self.endian is not None:
            configuration_json["endian"] = self.endian
        return {**super().to_json(), "configuration": configuration_json}

    def validate(self, array_metadata: ArrayMetadata) -> None:
        assert (
            not array_metadata.data_type.has_endianness or self.endian is not None
        ), "The `endian` configuration needs to be specified for multi-byte data types."

    def _get_byteorder(self, array: np.ndarray) -> Endian:
        if array.dtype.byteorder == "<":
            return "little"
        elif array.dtype.byteorder == ">":
            return "big"
        else:
            import sys

            return sys.byteorder

    async def decode(
        self,
        chunk_bytes: BytesLike,
        chunk_metadata: ChunkMetadata,
    ) -> np.ndarray:
        if chunk_metadata.dtype.itemsize > 0:
            if self.endian == "little":
                prefix = "<"
            else:
                prefix = ">"
            dtype = np.dtype(f"{prefix}{chunk_metadata.data_type.to_numpy_shortname()}")
        else:
            dtype = np.dtype(f"|{chunk_metadata.data_type.to_numpy_shortname()}")
        chunk_array = np.frombuffer(chunk_bytes, dtype)

        # ensure correct chunk shape
        if chunk_array.shape != chunk_metadata.chunk_shape:
            chunk_array = chunk_array.reshape(
                chunk_metadata.chunk_shape,
            )
        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
        _chunk_metadata: ChunkMetadata,
    ) -> Optional[BytesLike]:
        if chunk_array.dtype.itemsize > 1:
            byteorder = self._get_byteorder(chunk_array)
            if self.endian != byteorder:
                new_dtype = chunk_array.dtype.newbyteorder(self.endian)
                chunk_array = chunk_array.astype(new_dtype)
        return chunk_array.tobytes()

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length


register_codec("bytes", BytesCodec)

# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
