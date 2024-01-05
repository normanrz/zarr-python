from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Tuple,
)

import numpy as np
from dataclasses import dataclass, replace

from zarr.v3.abc.codec import ArrayArrayCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import JSON

if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata


@dataclass(frozen=True)
class TransposeCodec(ArrayArrayCodec):
    order: Tuple[int, ...]

    name: Literal["transpose"] = "transpose"
    is_fixed_size = True

    def validate_evolve(self, chunk_metadata: ChunkMetadata) -> TransposeCodec:
        # Compatibility with older version of ZEP1
        if self.order == "F":  # type: ignore
            order = tuple(chunk_metadata.ndim - x - 1 for x in range(chunk_metadata.ndim))

        elif self.order == "C":  # type: ignore
            order = tuple(range(chunk_metadata.ndim))

        else:
            assert len(self.order) == chunk_metadata.ndim, (
                "The `order` tuple needs have as many entries as "
                + f"there are dimensions in the array. Got: {self.order}"
            )
            assert len(self.order) == len(set(self.order)), (
                "There must not be duplicates in the `order` tuple. " + f"Got: {self.order}"
            )
            assert all(0 <= x < chunk_metadata.ndim for x in self.order), (
                "All entries in the `order` tuple must be between 0 and "
                + f"the number of dimensions in the array. Got: {self.order}"
            )
            order = tuple(self.order)

        if order != self.order:
            return replace(self, order=order)
        return self

    def to_json(self) -> JSON:
        return {**super().to_json(), "configuration": {"order": self.order}}

    @classmethod
    def from_json(cls, val: JSON) -> TransposeCodec:
        assert val["name"] == cls.name
        order = val["configuration"]["order"]
        if isinstance(order, list):
            order = tuple(order)
        return cls(order=order)

    def resolve_metadata(self, chunk_metadata: ChunkMetadata) -> ChunkMetadata:
        from zarr.v3.metadata import ChunkMetadata

        return ChunkMetadata(
            chunk_shape=tuple(
                chunk_metadata.chunk_shape[self.order[i]] for i in range(chunk_metadata.ndim)
            ),
            data_type=chunk_metadata.data_type,
            fill_value=chunk_metadata.fill_value,
            runtime_configuration=chunk_metadata.runtime_configuration,
        )

    async def decode(self, chunk_array: np.ndarray, chunk_metadata: ChunkMetadata) -> np.ndarray:
        inverse_order = [0 for _ in range(chunk_metadata.ndim)]
        for x, i in enumerate(self.order):
            inverse_order[x] = i
        chunk_array = chunk_array.transpose(inverse_order)
        return chunk_array

    async def encode(
        self, chunk_array: np.ndarray, chunk_metadata: ChunkMetadata
    ) -> Optional[np.ndarray]:
        chunk_array = chunk_array.transpose(self.order)
        return chunk_array

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length


register_codec("transpose", TransposeCodec)
