"""Request log: every served request, its routing decision, answer and feedback.

Online monitoring (Phase 6) and the retraining trigger (Phase 7) read from this table.
Postgres in Docker; any SQLAlchemy URL works (tests use SQLite).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    inspect,
    select,
    update,
)

metadata = MetaData()

requests_table = Table(
    "requests",
    metadata,
    Column("request_id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("input", Text, nullable=False),
    Column("output", Text, nullable=False),
    Column("adapter", String(32), nullable=False, index=True),  # task name or "base"
    Column("adapter_version", String(16)),
    Column("router_version", String(16)),
    Column("route_label", String(32)),
    Column("route_confidence", Float),
    Column("route_reason", String(32)),
    Column("latency_ms", Float),
    Column("output_tokens", Integer),
    Column("quality_score", Float),  # reference-free online quality (monitoring/quality.py)
    Column("quality_checks", Text),  # JSON with the individual checks behind the score
    Column("feedback", String(8)),  # "up" | "down"
    Column("correction", Text),  # optional corrected answer from the user
)


def _sqlalchemy_url(url: str) -> str:
    # Plain postgresql:// URLs (shared with other tools) use the psycopg 3 driver here.
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


class RequestStore:
    def __init__(self, url: str):
        self.engine = create_engine(_sqlalchemy_url(url), pool_pre_ping=True)

    def init_schema(self) -> None:
        """Create the table, and add columns introduced after it was first created."""
        metadata.create_all(self.engine)
        existing = {c["name"] for c in inspect(self.engine).get_columns("requests")}
        with self.engine.begin() as conn:
            for column in requests_table.columns:
                if column.name not in existing:
                    ddl = column.type.compile(dialect=self.engine.dialect)
                    conn.exec_driver_sql(f"ALTER TABLE requests ADD COLUMN {column.name} {ddl}")

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            return True
        except Exception:
            return False

    def log_request(self, **row: Any) -> None:
        row.setdefault("created_at", datetime.now(UTC))
        with self.engine.begin() as conn:
            conn.execute(insert(requests_table).values(**row))

    def add_feedback(self, request_id: str, rating: str, correction: str | None) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(requests_table)
                .where(requests_table.c.request_id == request_id)
                .values(feedback=rating, correction=correction)
            )
        return result.rowcount > 0

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(requests_table).where(requests_table.c.request_id == request_id)
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def recent(
        self, adapter: str, version: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Most recent requests served by ``adapter`` (optionally one version), newest first."""
        query = select(requests_table).where(requests_table.c.adapter == adapter)
        if version is not None:
            query = query.where(requests_table.c.adapter_version == str(version))
        query = query.order_by(requests_table.c.created_at.desc()).limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
        return [dict(r) for r in rows]
