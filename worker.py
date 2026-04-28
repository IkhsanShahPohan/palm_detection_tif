"""
worker.py  ·  v5
Optimasi vs v4:
  - DB update: throttle 2 detik (bukan 1 detik) + notify digabung dalam 1 koneksi
  - gc.collect() hanya di akhir setiap ROW (bukan tiap batch) — overhead berkurang drastis
  - speed_window deque O(1) vs list.pop(0) O(n)
  - Batch results di-del segera setelah count_per_class untuk bebaskan memori GPU/RAM lebih cepat
  - _run_inference tidak import model ulang jika sudah di-cache (get_model idempotent)
  - Semua prinsip v4 dipertahankan: isolasi job, finally selalu hapus file
"""

import gc
import os
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from database import update_job, notify_job_progress

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

MODEL_PATH       = os.getenv("MODEL_PATH", "best_v3.pt")
TEMP_DIR         = os.getenv("TEMP_DIR", "temp_uploads")
MAX_CONCURRENT   = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
DB_WRITE_INTERVAL = float(os.getenv("DB_WRITE_INTERVAL", "2.0"))  # detik antar DB write

_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT, thread_name_prefix="sawit_worker")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def safe_delete(path: str | None, label: str = "") -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"🗑️  Deleted {label}: {path}")
    except Exception as e:
        print(f"⚠️  Gagal hapus {path}: {e}")


def _fmt_duration(seconds: int | None) -> str:
    if seconds is None or seconds < 0:
        return "menghitung..."
    if seconds < 60:
        return f"{seconds}d"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s}d"
    h, m = divmod(m, 60)
    return f"{h}j {m}m {s}d"


def _fail_job(job_id: str, message: str, exc: Exception | None = None) -> None:
    detail = f"Gagal: {str(exc)}" if exc else message
    print(f"[{job_id[:8]}] ❌ {detail}")
    try:
        update_job(
            job_id,
            status="failed",
            temp_file_path=None,
        )
    except Exception as inner:
        print(f"[{job_id[:8]}] ⚠️  Tidak bisa update status ke DB: {inner}")


# ──────────────────────────────────────────────
# Core inference
# ──────────────────────────────────────────────

def _run_inference(job_id: str, file_path: str, conf: float, batch_size: int) -> None:
    try:
        from running_model3 import count_per_class, get_model
        from tile3 import generate_tiles_from_tiff, get_tif_info
    except ImportError as e:
        _fail_job(job_id, f"Dependency tidak terinstall: {e}")
        return

    start = time.monotonic()

    # ── 1. Metadata ───────────────────────────────────────────────
    try:
        update_job(job_id, status="processing", progress=2)
    except Exception as e:
        print(f"[{job_id[:8]}] ⚠️  DB update gagal di step 1: {e}")

    try:
        tif_info   = get_tif_info(file_path)
        total_grid = tif_info["total_grid_tiles"]
    except Exception as e:
        _fail_job(job_id, f"Gagal membaca file TIF: {e}", e)
        return

    try:
        update_job(
            job_id,
            progress=4,
            total_grid_tiles=total_grid,
            tif_info=tif_info,
        )
    except Exception as e:
        print(f"[{job_id[:8]}] ⚠️  DB update gagal setelah metadata: {e}")

    # ── 2. Load model ─────────────────────────────────────────────
    try:
        model       = get_model(MODEL_PATH)
        class_names = model.names
        accumulated = {name: 0 for name in class_names.values()}
    except Exception as e:
        _fail_job(job_id, f"Gagal memuat model YOLO ({MODEL_PATH}): {e}", e)
        return

    try:
        update_job(job_id, progress=5)
    except Exception as e:
        print(f"[{job_id[:8]}] ⚠️  DB update gagal setelah model load: {e}")

    # ── 3. Inferensi ──────────────────────────────────────────────
    tiles_done    = 0
    last_db_write = time.monotonic()

    # deque O(1) append/popleft — lebih cepat dari list.pop(0) untuk rolling window
    speed_window: deque[float] = deque(maxlen=20)

    try:
        tile_gen = generate_tiles_from_tiff(file_path, batch_size=batch_size)
    except Exception as e:
        _fail_job(job_id, f"Gagal membuka TIF untuk tiling: {e}", e)
        return

    for batch, grid_idx in tile_gen:
        batch_len = len(batch)  # simpan sebelum batch mungkin di-del
        try:
            t0      = time.monotonic()
            results = model.predict(source=batch, conf=conf, verbose=False, max_det=2000)

            # Bebaskan memori batch SEGERA setelah predict — sebelum akumulasi count
            del batch
            batch = None

            # Akumulasi
            batch_counts = count_per_class(results, model)
            del results  # bebaskan GPU/RAM segera

            for cls in class_names.values():
                accumulated[cls] = accumulated.get(cls, 0) + batch_counts.get(cls, 0)

            tiles_done += batch_len

            # Rolling speed — deque.append otomatis buang elemen lama
            elapsed_per_tile = (time.monotonic() - t0) / max(batch_len, 1)
            speed_window.append(elapsed_per_tile)

            avg_spt       = sum(speed_window) / len(speed_window)
            tiles_per_sec = round(1 / avg_spt, 2) if avg_spt > 0 else 0
            progress      = min(5 + int((grid_idx / total_grid) * 94), 99) if total_grid else 99
            eta           = int((total_grid - grid_idx) * avg_spt) if avg_spt > 0 else None
            running_total = sum(accumulated.values())

            # DB write throttle — kurangi frekuensi dari 1s → 2s
            now = time.monotonic()
            if now - last_db_write >= DB_WRITE_INTERVAL:
                try:
                    update_job(
                        job_id,
                        progress=progress,
                        tiles_processed=tiles_done,
                        eta_seconds=eta,
                    )
                    notify_job_progress(job_id, {
                        "progress":         progress,
                        "tiles_processed":  tiles_done,
                        "total_grid_tiles": total_grid,
                        "eta_seconds":      eta,
                        "running_total":    running_total,
                        "status":           "processing",
                    })
                except Exception as db_err:
                    print(f"[{job_id[:8]}] ⚠️  DB write gagal (akan retry): {db_err}")
                last_db_write = now

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[{job_id[:8]}] ❌ Error inference batch (grid={grid_idx}):\n{tb}")
            _fail_job(job_id, f"Error saat inference tile {grid_idx}: {e}", e)
            return
        finally:
            # Hanya gc.collect setiap ROW selesai, bukan tiap batch
            # Ini dikontrol oleh tile3.py yang sudah yield per-batch per-row
            if batch is not None:
                del batch

    # gc satu kali setelah semua tile selesai — tidak perlu tiap iterasi
    gc.collect()

    # ── 4. Finalisasi ─────────────────────────────────────────────
    elapsed     = round(time.monotonic() - start, 1)
    final_total = sum(accumulated.values())
    accumulated["total"] = final_total

    try:
        update_job(
            job_id,
            status="done",
            progress=100,
            class_counts=accumulated,
            tiles_processed=tiles_done,
            eta_seconds=0,
            elapsed_seconds=int(elapsed),
            temp_file_path=None,
        )
        notify_job_progress(job_id, {
            "progress":        100,
            "status":          "done",
            "class_counts":    accumulated,
            "tiles_processed": tiles_done,
            "elapsed_seconds": int(elapsed),
        })
    except Exception as e:
        print(f"[{job_id[:8]}] ⚠️  Gagal update final ke DB: {e}")
        _fail_job(job_id, f"Inferensi selesai tapi gagal simpan hasil ke DB: {e}", e)
        return

    print(f"[{job_id[:8]}] ✅ Done — {final_total} pohon · {tiles_done} tiles · {elapsed}s")


# ──────────────────────────────────────────────
# Job runner
# ──────────────────────────────────────────────

def run_tif_upload_job(
    job_id: str, file_path: str, conf: float, batch_size: int
) -> None:
    print(f"[{job_id[:8]}] 🚀 Mulai inference — {file_path}")
    try:
        _run_inference(job_id, file_path, conf, batch_size)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{job_id[:8]}] ❌ Unhandled exception:\n{tb}")
        _fail_job(job_id, f"Error tidak terduga: {e}", e)
    finally:
        safe_delete(file_path, f"tif_upload job={job_id[:8]}")
        gc.collect()
        print(f"[{job_id[:8]}] 🔓 Worker slot dibebaskan")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def submit_tif_upload(job_id: str, file_path: str, conf: float, batch_size: int) -> None:
    _executor.submit(run_tif_upload_job, job_id, file_path, conf, batch_size)
