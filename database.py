"""
database.py
Koneksi PostgreSQL Supabase via psycopg2 ThreadedConnectionPool.
Pool memastikan banyak thread (worker inference + FastAPI request handler)
bisa baca/tulis DB bersamaan tanpa konflik.

Fitur tambahan:
- notify_job_progress: kirim pg_notify untuk Supabase Realtime
- get_history: filter date_from, date_to, sort_by
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "")

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL belum diset. Salin .env.example ke .env lalu isi nilai DATABASE_URL."
            )
        _pool = psycopg2.pool.ThreadedConnectionPool(2, 10, DATABASE_URL)
    return _pool


@contextmanager
def get_conn():
    """
    Context manager — ambil koneksi dari pool, commit/rollback otomatis,
    kembalikan ke pool setelah selesai. Thread-safe.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ─────────────────────────────────────────────
# DDL — tabel dibuat otomatis saat startup
# ─────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    user_id           TEXT        NOT NULL DEFAULT 'anonymous',
    source_type       TEXT        NOT NULL DEFAULT 'tif_upload',
    status            TEXT        NOT NULL DEFAULT 'queued',
    progress          INTEGER     NOT NULL DEFAULT 0,
    total_detected    INTEGER     NOT NULL DEFAULT 0,
    class_counts      JSONB       NOT NULL DEFAULT '{}',
    total_grid_tiles  INTEGER     NOT NULL DEFAULT 0,
    tiles_processed   INTEGER     NOT NULL DEFAULT 0,
    eta_seconds       INTEGER,
    elapsed_seconds   INTEGER,
    tif_info          JSONB       NOT NULL DEFAULT '{}',
    filename          TEXT,
    file_size_bytes   BIGINT,
    temp_file_path    TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Migrasi: tambah kolom baru jika tabel sudah ada dari versi sebelumnya
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS elapsed_seconds INTEGER;
ALTER TABLE jobs DROP COLUMN IF EXISTS tiles_per_second;
ALTER TABLE jobs DROP COLUMN IF EXISTS message;

CREATE INDEX IF NOT EXISTS idx_jobs_user_id    ON jobs (user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status     ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at DESC);
"""

def init_db():
    """Panggil sekali saat startup — buat tabel + index jika belum ada."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
    print("✅ Database siap")


# ─────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────

def create_job(
    job_id: str,
    user_id: str,
    source_type: str,
    filename: str | None = None,
    file_size_bytes: int | None = None,
    temp_file_path: str | None = None,
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs
                    (job_id, user_id, source_type, filename,
                     file_size_bytes, temp_file_path)
                VALUES (%s,%s,%s,%s,%s,%s)
                """,
                (job_id, user_id, source_type, filename,
                 file_size_bytes, temp_file_path),
            )


_UPDATABLE = {
    "status", "progress",
    "total_detected", "class_counts",
    "total_grid_tiles", "tiles_processed",
    "eta_seconds", "elapsed_seconds",
    "tif_info", "temp_file_path",
}


def update_job(job_id: str, **kwargs) -> None:
    """Update satu atau banyak kolom sekaligus. Aman dipanggil dari thread manapun."""
    cols, vals = [], []
    for k, v in kwargs.items():
        if k not in _UPDATABLE:
            continue
        if isinstance(v, dict):
            v = json.dumps(v)
        cols.append(f"{k} = %s")
        vals.append(v)

    if not cols:
        return

    cols.append("updated_at = NOW()")
    vals.append(job_id)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE jobs SET {', '.join(cols)} WHERE job_id = %s",
                tuple(vals),
            )


def get_job(job_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
    return _normalize(dict(row)) if row else None


def get_history(
    user_id: str | None = None,
    status: str | None = None,
    source_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    conds, params = ["1=1"], []

    if user_id:
        conds.append("user_id = %s"); params.append(user_id)
    if status:
        conds.append("status = %s"); params.append(status)
    if source_type:
        conds.append("source_type = %s"); params.append(source_type)
    if date_from:
        conds.append("created_at >= %s"); params.append(date_from)
    if date_to:
        conds.append("created_at <= (%s::date + INTERVAL '1 day')"); params.append(date_to)

    allowed_sort = {"created_at", "updated_at", "total_detected", "progress"}
    order_col    = sort_by if sort_by in allowed_sort else "created_at"
    order_dir    = "ASC" if sort_order.lower() == "asc" else "DESC"

    where = " AND ".join(conds)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM jobs WHERE {where}", tuple(params))
            total = cur.fetchone()["n"]
            cur.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where}
                ORDER BY {order_col} {order_dir}
                LIMIT %s OFFSET %s
                """,
                tuple(params) + (limit, offset),
            )
            rows = [_normalize(dict(r)) for r in cur.fetchall()]

    return {"total": total, "limit": limit, "offset": offset, "data": rows}


def count_active_jobs(user_id: str) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM jobs WHERE user_id=%s AND status IN ('queued','processing')",
                (user_id,),
            )
            return cur.fetchone()[0]


def get_stale_jobs() -> list[dict]:
    """Ambil job masih queued/processing — untuk cleanup saat server restart."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT job_id, temp_file_path FROM jobs WHERE status IN ('queued','processing')"
            )
            return [dict(r) for r in cur.fetchall()]


# ─────────────────────────────────────────────
# Supabase Realtime — pg_notify
# ─────────────────────────────────────────────

def notify_job_progress(job_id: str, payload: dict) -> None:
    try:
        payload["job_id"] = job_id
        payload_str = json.dumps(payload)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_notify('job_progress', %s)",
                    (payload_str,),
                )
    except Exception as e:
        print(f"[notify] ⚠️  pg_notify gagal untuk job {job_id[:8]}: {e}")


# ─────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────

def _normalize(row: dict) -> dict:
    for col in ("class_counts", "tif_info"):
        v = row.get(col)
        if isinstance(v, str):
            try:
                row[col] = json.loads(v)
            except Exception:
                row[col] = {}
        elif v is None:
            row[col] = {}
    for col in ("created_at", "updated_at"):
        v = row.get(col)
        if isinstance(v, datetime):
            row[col] = v.isoformat()
    return row
