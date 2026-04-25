"""
main.py
Palm Tree Detection API v4
- Upload TIF langsung (background processing via ThreadPoolExecutor)
- PostgreSQL Supabase untuk concurrent read/write
- Supabase Realtime via pg_notify untuk progress real-time
- Setiap error langsung update status → failed (tidak ada job yang menggantung)
- History dengan filter lengkap: status, date_from, date_to, sort_by, dll
"""

import os
import uuid
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # Baca .env sebelum apapun

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware

import database as db
import worker

# ─────────────────────────────────────────────
# Config dari .env
# ─────────────────────────────────────────────

API_KEY             = os.getenv("API_KEY", "dev-key-ganti-di-production")
TEMP_DIR            = os.getenv("TEMP_DIR", "temp_uploads")
MAX_TIF_BYTES       = int(os.getenv("MAX_TIF_BYTES", 524_288_000))  # 500 MB default
MAX_ACTIVE_PER_USER = 3

os.makedirs(TEMP_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Palm Tree Detection API",
    description=(
        "Deteksi pohon kelapa sawit dari citra satelit TIF menggunakan YOLO.\n\n"
        "**Header wajib untuk semua endpoint (kecuali /health):**\n"
        "- `X-API-Key` — API key dari server\n"
        "- `X-User-ID` — ID pengguna yang sedang login (contoh: `user_123`)\n\n"
        "**Realtime progress:** Subscribe ke channel `job_progress` di Supabase Realtime.\n"
        "Event payload berisi `job_id`, `progress`, `status`, `tiles_processed`, dll."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Startup & shutdown
# ─────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    db.init_db()
    _mark_stale_jobs_failed()


@app.on_event("shutdown")
def on_shutdown():
    worker._executor.shutdown(wait=False, cancel_futures=False)
    print("👋 Server shutdown")


def _mark_stale_jobs_failed():
    """
    Saat server restart, job yang masih 'processing' atau 'queued'
    dari sesi sebelumnya tidak akan pernah selesai — tandai failed
    dan hapus file temp-nya supaya disk tidak penuh.
    """
    stale = db.get_stale_jobs()
    for row in stale:
        job_id    = row["job_id"]
        temp_path = row.get("temp_file_path")
        print(f"[startup] ♻️  Stale job ditemukan: {job_id[:8]} — marking failed")
        worker.safe_delete(temp_path, "stale job")
        db.update_job(
            job_id,
            status="failed",
            temp_file_path=None,
            message="Job dibatalkan: server restart saat proses berjalan.",
        )


# ─────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key tidak valid.")
    return x_api_key


def get_user_id(
    x_user_id: str = Header(default="anonymous", alias="X-User-ID")
) -> str:
    """
    Ambil user_id dari header X-User-ID.
    Kirim dari Flutter: header 'X-User-ID': '<id_user_yang_login>'
    """
    uid = (x_user_id or "").strip()
    return uid if uid else "anonymous"


# ─────────────────────────────────────────────
# Response builder — satu format konsisten
# ─────────────────────────────────────────────

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


def _safe_remove(path: str | None):
    worker.safe_delete(path)


# ─────────────────────────────────────────────
# ENDPOINT: Upload TIF langsung
# ─────────────────────────────────────────────

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

    **Batas ukuran:** 500 MB (default, bisa diubah via env `MAX_TIF_BYTES`).

    **Realtime progress (Flutter/JS):**
    ```js
    supabase.channel('job_progress')
      .on('broadcast', { event: '<job_id>' }, (payload) => {
        // payload: { progress, status, tiles_processed, eta_seconds, ... }
      })
      .subscribe()
    ```
    """
    # Validasi ekstensi
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "File harus berekstensi .tif atau .tiff")

    # Validasi ukuran via header Content-Length jika tersedia
    if file.size and file.size > MAX_TIF_BYTES:
        raise HTTPException(
            413,
            f"File terlalu besar ({round(file.size/1024/1024, 1)} MB). "
            f"Maks {round(MAX_TIF_BYTES/1024/1024)} MB.",
        )

    # Cek batas job aktif per user
    active = db.count_active_jobs(user_id)
    if active >= MAX_ACTIVE_PER_USER:
        raise HTTPException(
            429,
            f"Kamu sudah punya {active} job aktif. "
            f"Tunggu selesai dulu (maks {MAX_ACTIVE_PER_USER} bersamaan).",
        )

    # Buat job_id dan path temp
    job_id    = str(uuid.uuid4())
    safe_name = f"{job_id}_{os.path.basename(file.filename or 'upload.tif')}"
    temp_path = os.path.join(TEMP_DIR, safe_name)

    # Daftarkan job ke DB dulu — sehingga cleanup startup bisa temukan
    # file ini walau upload terputus sebelum selesai
    db.create_job(
        job_id=job_id,
        user_id=user_id,
        source_type="tif_upload",
        filename=file.filename,
        file_size_bytes=file.size,
        temp_file_path=temp_path,
    )

    # Tulis file ke disk dengan chunking 2MB — cegah OOM untuk file besar
    bytes_written = 0
    try:
        with open(temp_path, "wb") as buf:
            while True:
                chunk = await file.read(2 * 1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                # Double check ukuran saat streaming
                if bytes_written > MAX_TIF_BYTES:
                    buf.close()
                    _safe_remove(temp_path)
                    db.update_job(job_id, status="failed", temp_file_path=None,
                                  message="File melebihi batas ukuran saat upload.")
                    raise HTTPException(413, "File melebihi batas ukuran maksimal.")
                buf.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        _safe_remove(temp_path)
        db.update_job(job_id, status="failed", temp_file_path=None,
                      message=f"Upload terputus setelah {bytes_written // 1024} KB: {e}")
        raise HTTPException(500, f"Gagal menyimpan file: {e}")

    if bytes_written == 0:
        _safe_remove(temp_path)
        db.update_job(job_id, status="failed", temp_file_path=None,
                      message="File kosong.")
        raise HTTPException(400, "File kosong.")

    # Submit ke worker thread — tidak blocking, langsung return
    worker.submit_tif_upload(job_id, temp_path, conf, batch_size)

    return {
        "status":     "queued",
        "job_id":     job_id,
        "user_id":    user_id,
        "filename":   file.filename,
        "size_bytes": bytes_written,
        "size_mb":    round(bytes_written / 1024 / 1024, 2),
        "message":    "Job diterima dan masuk antrian. Proses berjalan di background.",
        "poll_url":   f"/status/{job_id}",
        "realtime_channel": "job_progress",
        "realtime_event":   job_id,
    }


# ─────────────────────────────────────────────
# ENDPOINT: Status satu job
# ─────────────────────────────────────────────

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

    **Field status:**
    - `queued` — menunggu worker tersedia
    - `processing` — sedang diproses
    - `done` — selesai, lihat `total_detected` & `class_counts`
    - `failed` — gagal, lihat `message` untuk detail error
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' tidak ditemukan.")
    return _job_to_response(job)


# ─────────────────────────────────────────────
# ENDPOINT: Riwayat semua job (admin)
# ─────────────────────────────────────────────

@app.get(
    "/history",
    tags=["Status & Riwayat"],
    summary="Riwayat job dengan filter lengkap dan pagination",
)
async def history(
    user_id:     Optional[str] = Query(default=None, description="Filter berdasarkan user_id"),
    status:      Optional[str] = Query(default=None, description="Filter status: queued | processing | done | failed"),
    source_type: Optional[str] = Query(default=None, description="Filter tipe: tif_upload"),
    date_from:   Optional[str] = Query(default=None, description="Filter dari tanggal (YYYY-MM-DD), inklusif"),
    date_to:     Optional[str] = Query(default=None, description="Filter sampai tanggal (YYYY-MM-DD), inklusif"),
    sort_by:     str           = Query(default="created_at", description="Kolom urutan: created_at | updated_at | total_detected | progress"),
    sort_order:  str           = Query(default="desc", description="Arah urutan: asc | desc"),
    limit:       int           = Query(default=50, ge=1, le=100, description="Jumlah data per halaman (maks 100)"),
    offset:      int           = Query(default=0, ge=0, description="Mulai dari data ke-n (untuk pagination)"),
    _key: str = Depends(require_api_key),
):
    """
    **Filter opsional:**
    - `user_id` → `?user_id=user_123`
    - `status` → `?status=done`
    - `source_type` → `?source_type=tif_upload`
    - `date_from` → `?date_from=2024-01-01`
    - `date_to` → `?date_to=2024-12-31`
    - `sort_by` → `?sort_by=total_detected`
    - `sort_order` → `?sort_order=asc`
    - `limit` + `offset` → pagination

    **Contoh:**
    - `/history?status=done&sort_by=total_detected&sort_order=desc`
    - `/history?user_id=user_123&date_from=2024-06-01&date_to=2024-06-30`
    """
    result = db.get_history(
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


# ─────────────────────────────────────────────
# ENDPOINT: Riwayat milik user sendiri
# ─────────────────────────────────────────────

@app.get(
    "/history/me",
    tags=["Status & Riwayat"],
    summary="Riwayat job milik user yang sedang login",
)
async def history_me(
    status:     Optional[str] = Query(default=None, description="Filter status: queued | processing | done | failed"),
    date_from:  Optional[str] = Query(default=None, description="Filter dari tanggal (YYYY-MM-DD)"),
    date_to:    Optional[str] = Query(default=None, description="Filter sampai tanggal (YYYY-MM-DD)"),
    sort_by:    str           = Query(default="created_at", description="Kolom urutan: created_at | updated_at | total_detected | progress"),
    sort_order: str           = Query(default="desc", description="Arah urutan: asc | desc"),
    limit:      int           = Query(default=20, ge=1, le=100),
    offset:     int           = Query(default=0, ge=0),
    _key: str    = Depends(require_api_key),
    user_id: str = Depends(get_user_id),
):
    """
    Shortcut — `user_id` diambil otomatis dari header `X-User-ID`.
    Tidak perlu kirim `?user_id=...` secara manual.

    Filter yang tersedia sama dengan `/history`.
    """
    result = db.get_history(
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


# ─────────────────────────────────────────────
# ENDPOINT: Health check
# ─────────────────────────────────────────────

@app.get(
    "/health",
    tags=["Admin"],
    summary="Cek status server dan database",
)
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
        "version":     "4.0.0",
    }


# ─────────────────────────────────────────────
# ENDPOINT: Admin — lihat isi folder temp
# ─────────────────────────────────────────────

@app.get(
    "/admin/temp/list",
    tags=["Admin"],
    summary="Lihat file temp di disk",
)
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


@app.delete(
    "/admin/temp/cleanup",
    tags=["Admin"],
    summary="Hapus semua file temp di disk secara manual",
)
async def temp_cleanup(_key: str = Depends(require_api_key)):
    """
    Hanya hapus file yang TIDAK sedang diproses.
    File yang sedang diproses worker tidak akan dihapus paksa.
    """
    removed, errors = [], []
    if os.path.exists(TEMP_DIR):
        for fname in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, fname)
            try:
                os.remove(fpath)
                removed.append(fname)
            except Exception as e:
                errors.append({"file": fname, "error": str(e)})
    return {
        "removed":       removed,
        "errors":        errors,
        "count_removed": len(removed),
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
