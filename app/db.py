"""Async SQLAlchemy engine and session factory. Migrations run via Alembic on startup."""
from contextlib import asynccontextmanager

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_conn():
    """A single dedicated connection, checked out for its whole lifetime.

    Unlike get_db()'s Session — which returns its connection to the pool on every
    commit() — this keeps ONE physical connection until the block exits. Required
    for SESSION-level Postgres advisory locks, which are bound to a connection and
    would otherwise strand on a pooled connection (a later unlock on a different
    connection silently no-ops). Reads the live AsyncSessionLocal bind so tests
    that redirect it to a throwaway engine are honoured.
    """
    async with AsyncSessionLocal.kw["bind"].connect() as conn:
        yield conn


async def run_migrations() -> None:
    """Run pending Alembic migrations. Called once on app startup."""
    import asyncio
    from alembic import command
    from alembic.config import Config

    def _migrate():
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
        command.upgrade(cfg, "head")

    await asyncio.to_thread(_migrate)
