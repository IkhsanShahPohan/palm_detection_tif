"""
main.py  ·  Palm Tree Detection API v5
Optimasi vs v4:
  - Upload TIF ditulis ke disk dengan aiofiles (truly async, tidak bloking event loop)
  - Validasi ekstensi & quota langsung sebelum baca file (fail-fast)
  - DB call di-wrap run_in_executor agar tidak bloking event loop FastAPI
  - Chunk size 4 MB untuk throughput upload maksimal
  - ORJSONResponse untuk serialisasi JSON lebih cepat
  - Lifespan handler (startup/shutdown) — tidak pakai on_event yang deprecated
  - uvloop dipakai hanya jika tersedia (Linux/Mac); Windows pakai asyncio default
"""

import asyncio
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import aiofiles
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

import database as db
import worker

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_KEY             = os.getenv("API_KEY", "dev-key-ganti-di-production")
TEMP_DIR            = os.getenv("TEMP_DIR", "temp_uploads")
MAX_TIF_BYTES       = int(os.getenv("MAX_TIF_BYTES", 524_288_000))   # 500 MB
MAX_ACTIVE_PER_USER = 3
UPLOAD_CHUNK_SIZE   = int(os.getenv("UPLOAD_CHUNK_SIZE", 4 * 1024 * 1024))  # 4 MB

os.makedirs(TEMP_DIR, exist_ok=True)

# Executor khusus untuk operasi DB (psycopg2 sync) agar tidak bloking event loop
_db_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="db_thread")


# ──────────────────────────────────────────────
# Helpers async DB
# ──────────────────────────────────────────────

async def _db(fn, *args, **kwargs):
    """
    Jalankan fungsi DB sync di thread pool agar tidak bloking event loop.
    Semua panggilan db.* dari async handler harus lewat helper ini.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_db_executor, lambda: fn(*args, **kwargs))


async def _mark_stale_jobs_failed():
    stale = await _db(db.get_stale_jobs)
    for row in stale:
        job_id    = row["job_id"]
        temp_path = row.get("temp_file_path")
        print(f"[startup] ♻️  Stale job: {job_id[:8]} → failed")
        worker.safe_delete(temp_path, "stale")
        await _db(
            db.update_job, job_id,
            status="failed",
            temp_file_path=None,
            message="Job dibatalkan: server restart saat proses berjalan.",
        )


# ──────────────────────────────────────────────
# Lifespan (startup + shutdown)
# Menggantikan @app.on_event yang deprecated sejak FastAPI 0.93
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    # ── Startup ───────────────────────────────
    await _db(db.init_db)
    await _mark_stale_jobs_failed()
    print("✅ Server siap")
    yield
    # ── Shutdown ──────────────────────────────
    worker._executor.shutdown(wait=False, cancel_futures=False)
    _db_executor.shutdown(wait=False)
    print("👋 Server shutdown")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Palm Tree Detection API",
    description=(
        "Deteksi pohon kelapa sawit dari citra satelit TIF menggunakan YOLO.\n\n"
        "**Header wajib untuk semua endpoint (kecuali /health):**\n"
        "- `X-API-Key` — API key dari server\n"
        "- `X-User-ID` — ID pengguna yang sedang login\n\n"
        "**Realtime progress:** Subscribe ke channel `job_progress` di Supabase Realtime."
    ),
    version="5.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key tidak valid.")
    return x_api_key


def get_user_id(
    x_user_id: str = Header(default="anonymous", alias="X-User-ID")
) -> str:
    uid = (x_user_id or "").strip()
    return uid if uid else "anonymous"


# ──────────────────────────────────────────────
# Response builder
# ──────────────────────────────────────────────

def _job_to_response(job: dict) -> dict:
    return {
        "job_id":           job["job_id"],
        "user_id":          job["user_id"],
        "source_type":      job["source_type"],
        "status":           job["status"],
        "progress":         job["progress"],
        "total_detected":   job["total_detected"],
        "class_counts":     job["class_counts"],
        "tiles_processed":  job["tiles_processed"],
        "total_grid_tiles": job["total_grid_tiles"],
        "tiles_per_second": job["tiles_per_second"],
        "eta_seconds":      job["eta_seconds"],
        "tif_info":         job["tif_info"],
        "filename":         job.get("filename"),
        "file_size_bytes":  job.get("file_size_bytes"),
        "message":          job["message"],
        "created_at":       job["created_at"],
        "updated_at":       job["updated_at"],
    }


# ──────────────────────────────────────────────
# ENDPOINT: Upload TIF
# ──────────────────────────────────────────────

@app.post(
    "/detect/tif",
    tags=["Deteksi"],
    summary="Upload file TIF untuk deteksi sawit",
)
async def detect_tif_upload(
    file: UploadFile = File(..., description="File .tif / .tiff"),
    conf: float      = Form(default=0.5, ge=0.1, le=1.0,
                            description="Confidence threshold (0.1–1.0)"),
    batch_size: int  = Form(default=4, ge=1, le=16,
                            description="Tile per batch inferensi"),
    _key: str        = Depends(require_api_key),
    user_id: str     = Depends(get_user_id),
):
    """
    Upload file TIF/TIFF — proses berjalan di **background** (non-blocking).

    Response langsung mengembalikan `job_id`.
    Gunakan `GET /status/{job_id}` untuk polling progress,
    atau subscribe ke Supabase Realtime channel `job_progress` untuk update real-time.

    **Batas ukuran:** 500 MB default (ubah via env `MAX_TIF_BYTES`).
    """

    # ── 1. Validasi cepat sebelum baca file (fail-fast) ───────────
    fname_lower = (file.filename or "").lower()
    if not (fname_lower.endswith(".tif") or fname_lower.endswith(".tiff")):
        raise HTTPException(400, "File harus berekstensi .tif atau .tiff")

    if file.size and file.size > MAX_TIF_BYTES:
        raise HTTPException(
            413,
            f"File terlalu besar ({round(file.size / 1024 / 1024, 1)} MB). "
            f"Maks {round(MAX_TIF_BYTES / 1024 / 1024)} MB.",
        )

    # ── 2. Cek quota user ─────────────────────────────────────────
    active = await _db(db.count_active_jobs, user_id)
    if active >= MAX_ACTIVE_PER_USER:
        raise HTTPException(
            429,
            f"Kamu sudah punya {active} job aktif. "
            f"Tunggu selesai dulu (maks {MAX_ACTIVE_PER_USER} bersamaan).",
        )

    # ── 3. Siapkan path ───────────────────────────────────────────
    job_id    = str(uuid.uuid4())
    safe_name = f"{job_id}_{os.path.basename(file.filename or 'upload.tif')}"
    temp_path = os.path.join(TEMP_DIR, safe_name)

    # ── 4. Daftarkan job ke DB (sebelum tulis file) ───────────────
    await _db(
        db.create_job,
        job_id=job_id,
        user_id=user_id,
        source_type="tif_upload",
        filename=file.filename,
        file_size_bytes=file.size,
        temp_file_path=temp_path,
    )

    # ── 5. Tulis file ke disk (truly async via aiofiles) ──────────
    bytes_written = 0
    try:
        async with aiofiles.open(temp_path, "wb") as buf:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_TIF_BYTES:
                    worker.safe_delete(temp_path)
                    await _db(
                        db.update_job, job_id,
                        status="failed",
                        temp_file_path=None,
                        message="File melebihi batas ukuran saat upload.",
                    )
                    raise HTTPException(
                        413,
                        f"File melebihi batas {round(MAX_TIF_BYTES / 1024 / 1024)} MB.",
                    )
                await buf.write(chunk)

    except HTTPException:
        raise

    except Exception as exc:
        worker.safe_delete(temp_path)
        await _db(
            db.update_job, job_id,
            status="failed",
            temp_file_path=None,
            message=f"Gagal menyimpan file: {exc}",
        )
        raise HTTPException(500, f"Gagal menyimpan file upload: {exc}")

    # ── 6. Validasi tidak kosong ──────────────────────────────────
    if bytes_written == 0:
        worker.safe_delete(temp_path)
        await _db(
            db.update_job, job_id,
            status="failed",
            temp_file_path=None,
            message="File kosong.",
        )
        raise HTTPException(400, "File tidak boleh kosong.")

    # ── 7. Update ukuran & submit ke worker ───────────────────────
    await _db(db.update_job, job_id, file_size_bytes=bytes_written)
    worker.submit_tif_upload(job_id, temp_path, conf, batch_size)

    return {
        "status":           "queued",
        "job_id":           job_id,
        "user_id":          user_id,
        "filename":         file.filename,
        "size_bytes":       bytes_written,
        "size_mb":          round(bytes_written / 1024 / 1024, 2),
        "message":          "Job diterima dan masuk antrian. Proses berjalan di background.",
        "poll_url":         f"/status/{job_id}",
        "realtime_channel": "job_progress",
        "realtime_event":   job_id,
    }


# ──────────────────────────────────────────────
# ENDPOINT: Status satu job
# ──────────────────────────────────────────────

@app.get(
    "/status/{job_id}",
    tags=["Status & Riwayat"],
    summary="Cek status dan progress job",
)
async def get_status(
    job_id: str,
    _key: str = Depends(require_api_key),
):
    """
    Polling status job. Panggil setiap 3–5 detik dari klien,
    atau gunakan Supabase Realtime untuk update instan tanpa polling.
    """
    job = await _db(db.get_job, job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' tidak ditemukan.")
    return _job_to_response(job)


# ──────────────────────────────────────────────
# ENDPOINT: Riwayat semua job
# ──────────────────────────────────────────────

@app.get(
    "/history",
    tags=["Status & Riwayat"],
    summary="Riwayat job dengan filter lengkap dan pagination",
)
async def history(
    user_id:     Optional[str] = Query(default=None, description="Filter berdasarkan user_id"),
    status:      Optional[str] = Query(default=None, description="queued | processing | done | failed"),
    source_type: Optional[str] = Query(default=None, description="tif_upload"),
    date_from:   Optional[str] = Query(default=None, description="YYYY-MM-DD, inklusif"),
    date_to:     Optional[str] = Query(default=None, description="YYYY-MM-DD, inklusif"),
    sort_by:     str           = Query(default="created_at", description="created_at | updated_at | total_detected | progress"),
    sort_order:  str           = Query(default="desc", description="asc | desc"),
    limit:       int           = Query(default=50, ge=1, le=100),
    offset:      int           = Query(default=0, ge=0),
    _key: str = Depends(require_api_key),
):
    result = await _db(
        db.get_history,
        user_id=user_id,
        status=status,
        source_type=source_type,
        date_from=date_from,
        date_to=date_to,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
    )
    return {
        "total":    result["total"],
        "limit":    result["limit"],
        "offset":   result["offset"],
        "has_more": (result["offset"] + len(result["data"])) < result["total"],
        "data":     [_job_to_response(j) for j in result["data"]],
    }


# ──────────────────────────────────────────────
# ENDPOINT: Riwayat milik user sendiri
# ──────────────────────────────────────────────

@app.get(
    "/history/me",
    tags=["Status & Riwayat"],
    summary="Riwayat job milik user yang sedang login",
)
async def history_me(
    status:     Optional[str] = Query(default=None),
    date_from:  Optional[str] = Query(default=None),
    date_to:    Optional[str] = Query(default=None),
    sort_by:    str           = Query(default="created_at"),
    sort_order: str           = Query(default="desc"),
    limit:      int           = Query(default=20, ge=1, le=100),
    offset:     int           = Query(default=0, ge=0),
    _key: str    = Depends(require_api_key),
    user_id: str = Depends(get_user_id),
):
    """user_id diambil otomatis dari header X-User-ID."""
    result = await _db(
        db.get_history,
        user_id=user_id,
        status=status,
        date_from=date_from,
        date_to=date_to,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
    )
    return {
        "user_id":  user_id,
        "total":    result["total"],
        "limit":    result["limit"],
        "offset":   result["offset"],
        "has_more": (result["offset"] + len(result["data"])) < result["total"],
        "data":     [_job_to_response(j) for j in result["data"]],
    }


# ──────────────────────────────────────────────
# ENDPOINT: Health
# ──────────────────────────────────────────────

@app.get("/health", tags=["Admin"], summary="Cek status server dan database")
async def health():
    """Tidak butuh API key — untuk monitoring uptime."""
    db_ok = False
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_ok = True
    except Exception as e:
        print(f"[health] DB error: {e}")

    return {
        "status":      "ok" if db_ok else "degraded",
        "database":    "connected" if db_ok else "error",
        "max_workers": worker.MAX_CONCURRENT,
        "version":     "5.0.0",
    }


# ──────────────────────────────────────────────
# ENDPOINT: Admin temp
# ──────────────────────────────────────────────

@app.get("/admin/temp/list", tags=["Admin"], summary="Lihat file temp di disk")
async def temp_list(_key: str = Depends(require_api_key)):
    files = []
    if os.path.exists(TEMP_DIR):
        for fname in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, fname)
            try:
                files.append({
                    "name":    fname,
                    "size_mb": round(os.path.getsize(fpath) / 1024 / 1024, 2),
                })
            except Exception:
                pass
    return {
        "dir":      TEMP_DIR,
        "count":    len(files),
        "total_mb": round(sum(f["size_mb"] for f in files), 2),
        "files":    files,
    }


@app.delete("/admin/temp/cleanup", tags=["Admin"], summary="Hapus semua file temp")
async def temp_cleanup(_key: str = Depends(require_api_key)):
    removed, errors = [], []
    if os.path.exists(TEMP_DIR):
        for fname in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, fname)
            try:
                os.remove(fpath)
                removed.append(fname)
            except Exception as e:
                errors.append({"file": fname, "error": str(e)})
    return {"removed": removed, "errors": errors, "count_removed": len(removed)}


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # uvloop tidak support Windows — deteksi platform otomatis
    is_windows = sys.platform.startswith("win")

    uvicorn_kwargs: dict = dict(
        app="main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        http="httptools",   # httptools support di Windows & Linux
        backlog=512,
    )

    if not is_windows:
        try:
            import uvloop  # noqa: F401
            uvicorn_kwargs["loop"] = "uvloop"
            print("🚀 Menggunakan uvloop (Linux/Mac)")
        except ImportError:
            print("ℹ️  uvloop tidak ditemukan, pakai asyncio default")
    else:
        print("ℹ️  Windows terdeteksi — menggunakan asyncio default (uvloop tidak support Windows)")

    uvicorn.run(**uvicorn_kwargs)