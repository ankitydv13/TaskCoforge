from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine , AsyncSession , async_sessionmaker
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = URL.create(
    "sqlite+aiosqlite",
    database=str(Path(__file__).resolve().parent.parent / "policy.db"),
)

engine = create_async_engine(
    url = DATABASE_URL,
    connect_args = {"check_same_thread":False},
)

#Making a async session
AsyncLocalSession = async_sessionmaker(
    engine,
    class_= AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncLocalSession() as db:
        yield db



