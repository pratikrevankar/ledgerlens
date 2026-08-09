"""asyncpg connection pool + pgvector registration."""
from __future__ import annotations

import os
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ledgerlens:ledgerlens@db:5432/ledgerlens")

_pool: Optional[asyncpg.Pool] = None


async def _init(conn: asyncpg.Connection) -> None:
    await register_vector(conn)  # lets us pass/receive python lists as vector()


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=8, init=_init)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
