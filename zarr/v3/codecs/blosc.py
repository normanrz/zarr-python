from __future__ import annotations
from functools import lru_cache

from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Optional,
)

import numcodecs
import numpy as np
from dataclasses import replace, dataclass
from numcodecs.blosc import Blosc

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import JSON, BytesLike, to_thread

if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata


BloscCNames = Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"]
BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]

# See https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
numcodecs.blosc.use_threads = False


blosc_shuffle_int_to_str: Dict[int, BloscShuffle] = {
    0: "noshuffle",
    1: "shuffle",
    2: "bitshuffle",
}


@dataclass(frozen=True)
class BloscCodec(BytesBytesCodec):
    typesize: int = 0
    cname: BloscCNames = "zstd"
    clevel: int = 5
    shuffle: BloscShuffle = "noshuffle"
    blocksize: int = 0

    name: Literal["blosc"] = "blosc"
    is_fixed_size = False

    def to_json(self) -> JSON:
        return {
            **super().to_json(),
            "configuration": {
                "typesize": self.typesize,
                "cname": self.cname,
                "clevel": self.clevel,
                "shuffle": self.shuffle,
                "blocksize": self.blocksize,
            },
        }

    def validate_evolve(self, chunk_metadata: ChunkMetadata) -> BloscCodec:
        new_codec = self
        if new_codec.typesize == 0:
            new_codec = replace(new_codec, typesize=chunk_metadata.data_type.byte_count)

        return new_codec

    @lru_cache
    def get_blosc_codec(self) -> Blosc:
        map_shuffle_str_to_int = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
        config_dict = {
            "cname": self.cname,
            "clevel": self.clevel,
            "shuffle": map_shuffle_str_to_int[self.shuffle],
            "blocksize": self.blocksize,
        }
        return Blosc.from_config(config_dict)

    async def decode(self, chunk_bytes: bytes, _chunk_metadata: ChunkMetadata) -> BytesLike:
        return await to_thread(self.get_blosc_codec().decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
        chunk_metadata: ChunkMetadata,
    ) -> Optional[BytesLike]:
        chunk_array = np.frombuffer(chunk_bytes, dtype=chunk_metadata.dtype)
        return await to_thread(self.get_blosc_codec().encode, chunk_array)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("blosc", BloscCodec)
