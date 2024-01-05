# Notes:
# 1. These are missing methods described in the spec. I expected to see these method definitions:
# def compute_encoded_representation_type(self, decoded_representation_type):
# def encode(self, decoded_value):
# def decode(self, encoded_value, decoded_representation_type):
# def partial_decode(self, input_handle, decoded_representation_type, decoded_regions):
# def compute_encoded_size(self, input_size):
# 2. Understand why array metadata is included on all codecs


from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Optional

import numpy as np

from zarr.v3.common import JSON, BytesLike, SliceSelection
from zarr.v3.store import StorePath


if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata


class Codec(ABC):
    name: str
    is_fixed_size: bool

    def to_json(self) -> JSON:
        return {"name": self.name}

    @classmethod
    def from_json(cls, val: JSON) -> Codec:
        assert val["name"] == cls.name
        return cls(**val.get("configuration", {}))

    @abstractmethod
    def compute_encoded_size(self, input_byte_length: int) -> int:
        pass

    def resolve_metadata(self, chunk_metadata: ChunkMetadata) -> ChunkMetadata:
        return chunk_metadata

    def validate_evolve(self, chunk_metadata: ChunkMetadata) -> Codec:
        return self


class ArrayArrayCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: np.ndarray,
        chunk_metadata: ChunkMetadata,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
        chunk_metadata: ChunkMetadata,
    ) -> Optional[np.ndarray]:
        pass


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
        chunk_metadata: ChunkMetadata,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
        chunk_metadata: ChunkMetadata,
    ) -> Optional[BytesLike]:
        pass


class ArrayBytesCodecPartialDecodeMixin:
    @abstractmethod
    async def decode_partial(
        self,
        store_path: StorePath,
        selection: SliceSelection,
        chunk_metadata: ChunkMetadata,
    ) -> Optional[np.ndarray]:
        pass


class ArrayBytesCodecPartialEncodeMixin:
    @abstractmethod
    async def encode_partial(
        self,
        store_path: StorePath,
        chunk_array: np.ndarray,
        selection: SliceSelection,
        chunk_metadata: ChunkMetadata,
    ) -> None:
        pass


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
        chunk_metadata: ChunkMetadata,
    ) -> BytesLike:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: BytesLike,
        chunk_metadata: ChunkMetadata,
    ) -> Optional[BytesLike]:
        pass
