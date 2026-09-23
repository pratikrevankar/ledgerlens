"""asyncpg connection pool + pgvector registration."""
from __future__ import annotations

import os
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ledgerlens:ledgerlens@db:5432/ledgerlens")

_pool: Optional[asyncpg.Pool] = None


async def _init(conn: asyncpg.Connection) -> None:
    # The vector type must exist before asyncpg can introspect it, and the pool's
    # very first connection runs before ingest creates the extension — so ensure it
    # here (idempotent) prior to registering the codec.
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector(conn)  # lets us pass/receive python lists as vector()


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        # Pool sizes are env-tunable: on serverless (Vercel) many function instances
        # each hold a pool, so keep max small and point DATABASE_URL at a pooled
        # endpoint (e.g. a pgbouncer/serverless-pooler DSN) to avoid exhausting the
        # database's connection limit.
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=int(os.getenv("DB_POOL_MIN", "1")),
            max_size=int(os.getenv("DB_POOL_MAX", "8")),
            init=_init,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
