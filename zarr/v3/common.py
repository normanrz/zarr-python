from __future__ import annotations

import asyncio
import contextvars
import functools
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"

BytesLike = Union[bytes, bytearray, memoryview]
ChunkCoords = Tuple[int, ...]
SliceSelection = Tuple[slice, ...]
Selection = Union[slice, SliceSelection]
JSON = Union[List["JSON"], Dict[str, "JSON"], int, float, str, bool, None]


def product(tup: ChunkCoords) -> int:
    return functools.reduce(lambda x, y: x * y, tup, 1)


T = TypeVar("T", bound=Tuple)
V = TypeVar("V")


async def concurrent_map(
    items: List[T], func: Callable[..., Awaitable[V]], limit: Optional[int] = None
) -> List[V]:
    if limit is None:
        return await asyncio.gather(*[func(*item) for item in items])

    else:
        sem = asyncio.Semaphore(limit)

        async def run(item):
            async with sem:
                return await func(*item)

        return await asyncio.gather(*[asyncio.ensure_future(run(item)) for item in items])


async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)


# Sentinel value
class _MISSING_TYPE:
    pass


MISSING = _MISSING_TYPE()
