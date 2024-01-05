from __future__ import annotations
from dataclasses import asdict, dataclass, field

import json
from asyncio import AbstractEventLoop
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from zarr.v3.abc.codec import Codec

from zarr.v3.common import JSON, ChunkCoords


@dataclass(frozen=True)
class RuntimeConfiguration:
    order: Literal["C", "F"] = "C"
    concurrency: Optional[int] = None
    asyncio_loop: Optional[AbstractEventLoop] = None


def runtime_configuration(
    order: Literal["C", "F"], concurrency: Optional[int] = None
) -> RuntimeConfiguration:
    return RuntimeConfiguration(order=order, concurrency=concurrency)


class DataType(Enum):
    bool = "bool"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    float32 = "float32"
    float64 = "float64"

    @property
    def byte_count(self) -> int:
        data_type_byte_counts = {
            DataType.bool: 1,
            DataType.int8: 1,
            DataType.int16: 2,
            DataType.int32: 4,
            DataType.int64: 8,
            DataType.uint8: 1,
            DataType.uint16: 2,
            DataType.uint32: 4,
            DataType.uint64: 8,
            DataType.float32: 4,
            DataType.float64: 8,
        }
        return data_type_byte_counts[self]

    def to_numpy_shortname(self) -> str:
        data_type_to_numpy = {
            DataType.bool: "bool",
            DataType.int8: "i1",
            DataType.int16: "i2",
            DataType.int32: "i4",
            DataType.int64: "i8",
            DataType.uint8: "u1",
            DataType.uint16: "u2",
            DataType.uint32: "u4",
            DataType.uint64: "u8",
            DataType.float32: "f4",
            DataType.float64: "f8",
        }
        return data_type_to_numpy[self]

    def to_json(self) -> JSON:
        return self.name

    @classmethod
    def from_json(cls, val: JSON) -> DataType:
        assert isinstance(val, str)
        return cls[val]


dtype_to_data_type = {
    "|b1": "bool",
    "bool": "bool",
    "|i1": "int8",
    "<i2": "int16",
    "<i4": "int32",
    "<i8": "int64",
    "|u1": "uint8",
    "<u2": "uint16",
    "<u4": "uint32",
    "<u8": "uint64",
    "<f4": "float32",
    "<f8": "float64",
}


@dataclass(frozen=True)
class _ExtensionPoint:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def to_json(self) -> JSON:
        return {"name": self.name, "configuration": asdict(self)}

    @classmethod
    def from_json(cls, val: JSON) -> _ExtensionPoint:
        raise NotImplementedError


@dataclass(frozen=True)
class ChunkGrid(_ExtensionPoint):
    @classmethod
    def from_json(cls, val: JSON) -> ChunkGrid:
        assert isinstance(val, dict)
        name = val["name"]
        if name == "regular":
            return RegularChunkGrid(chunk_shape=tuple(val["configuration"]["chunk_shape"]))
        else:
            raise NotImplementedError


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGrid):
    chunk_shape: ChunkCoords

    @property
    def name(self) -> Literal["regular"]:
        return "regular"


@dataclass(frozen=True)
class ChunkKeyEncoding(_ExtensionPoint):
    @classmethod
    def from_json(cls, val: JSON) -> ChunkKeyEncoding:
        assert isinstance(val, dict)
        name = val["name"]
        if name == "default":
            return DefaultChunkKeyEncoding(**val["configuration"])
        elif name == "v2":
            return V2ChunkKeyEncoding(**val["configuration"])
        else:
            raise NotImplementedError

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        raise NotImplementedError

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    separator: Literal[".", "/"] = "/"

    @property
    def name(self) -> Literal["default"]:
        return "default"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.separator.join(map(str, ("c",) + chunk_coords))


@dataclass(frozen=True)
class V2ChunkKeyEncoding(ChunkKeyEncoding):
    separator: Literal[".", "/"] = "."

    @property
    def name(self) -> Literal["v2"]:
        return "v2"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


class ShardingCodecIndexLocation(Enum):
    start = "start"
    end = "end"


@dataclass(frozen=True)
class ChunkMetadata:
    chunk_shape: ChunkCoords
    data_type: DataType
    fill_value: Any
    runtime_configuration: RuntimeConfiguration = runtime_configuration("C")

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)

    @property
    def ndim(self) -> int:
        return len(self.chunk_shape)


@dataclass(frozen=True)
class ArrayMetadata:
    shape: ChunkCoords
    data_type: DataType
    chunk_grid: ChunkGrid
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: Any
    codecs: List[Codec]
    attributes: Dict[str, JSON] = field(default_factory=dict)
    dimension_names: Optional[Tuple[str, ...]] = None
    zarr_format: Literal[3] = 3
    node_type: Literal["array"] = "array"

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_chunk_metadata(
        self, _chunk_coords: ChunkCoords, runtime_configuration: RuntimeConfiguration
    ) -> ChunkMetadata:
        assert isinstance(
            self.chunk_grid, RegularChunkGrid
        ), "Currently, only regular chunk grid is supported"
        return ChunkMetadata(
            chunk_shape=self.chunk_grid.chunk_shape,
            data_type=self.data_type,
            fill_value=self.fill_value,
            runtime_configuration=runtime_configuration,
        )

    def to_json(self) -> JSON:
        json = {
            "zarr_format": self.zarr_format,
            "node_type": self.node_type,
            "shape": self.shape,
            "data_type": self.data_type.name,
            "chunk_grid": self.chunk_grid.to_json(),
            "chunk_key_encoding": self.chunk_key_encoding.to_json(),
            "fill_value": self.fill_value,
            "codecs": [codec.to_json() for codec in self.codecs],
            "attributes": self.attributes,
        }

        if self.dimension_names is not None:
            json["dimension_names"] = self.dimension_names

        return json

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_json()).encode()

    @classmethod
    def from_json(cls, zarr_json: Any) -> ArrayMetadata:
        from zarr.v3.codecs.registry import from_json

        return ArrayMetadata(
            shape=tuple(zarr_json["shape"]),
            data_type=DataType.from_json(zarr_json["data_type"]),
            chunk_grid=ChunkGrid.from_json(zarr_json["chunk_grid"]),
            chunk_key_encoding=ChunkKeyEncoding.from_json(zarr_json["chunk_key_encoding"]),
            fill_value=zarr_json["fill_value"],
            codecs=[from_json(codec_json) for codec_json in zarr_json.get("codecs", [])],
            attributes=zarr_json["attributes"],
            dimension_names=tuple(zarr_json["dimension_names"])
            if "dimension_names" in zarr_json
            else None,
        )


def _parse_v2_fill_value(val: JSON) -> Union[None, int, float]:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    raise ValueError


@dataclass(frozen=True)
class ArrayV2Metadata:
    shape: ChunkCoords
    chunks: ChunkCoords
    dtype: np.dtype
    fill_value: Union[None, int, float] = 0
    order: Literal["C", "F"] = "C"
    filters: Optional[List[Dict[str, Any]]] = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: Optional[Dict[str, Any]] = None
    zarr_format: Literal[2] = 2

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_json(self) -> JSON:
        json = asdict(self)
        if self.dtype.fields is None:
            json["dtype"] = self.dtype.str
        else:
            json["dtype"] = self.dtype.descr
        return json

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_json()).encode()

    @classmethod
    def from_json(cls, zarr_json: Any) -> ArrayV2Metadata:
        return ArrayV2Metadata(
            shape=tuple(json["shape"]),
            chunks=tuple(json["chunks"]),
            dtype=np.dtype(json["dtype"]),
            fill_value=_parse_v2_fill_value(json.get("fill_value")),
            order=json.get("order", "C"),
            filters=json.get("filters"),
            dimension_separator=json.get("dimension_separator", "."),
            compressor=json.get("compressor"),
        )
