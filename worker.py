"""
worker.py
Semua logika berat: inference YOLO, manajemen file temp.

Prinsip utama:
- File temp SELALU dihapus di blok finally — tidak peduli sukses atau gagal
- Setiap exception (DB lost connection, rasterio error, dll) langsung update status → failed
- Setiap job terisolasi penuh — satu job gagal tidak mempengaruhi job lain
- pg_notify dipakai untuk Supabase Realtime progress updates
"""

import gc
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from database import update_job, notify_job_progress

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MODEL_PATH     = os.getenv("MODEL_PATH", "best_v3.pt")
TEMP_DIR       = os.getenv("TEMP_DIR", "temp_uploads")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))

# ThreadPoolExecutor: thread yang benar-benar menjalankan inference
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT, thread_name_prefix="sawit_worker")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def safe_delete(path: str | None, label: str = "") -> None:
    """Hapus file tanpa raise exception. Selalu dipanggil di finally."""
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
    """
    Helper terpusat untuk menandai job sebagai failed.
    Dipanggil dari mana saja — DB error, rasterio error, OOM, dll.
    Tidak pernah raise exception agar blok finally tetap berjalan.
    """
    detail = f"Gagal: {str(exc)}" if exc else message
    print(f"[{job_id[:8]}] ❌ {detail}")
    try:
        update_job(
            job_id,
            status="failed",
            temp_file_path=None,
            message=detail,
        )
    except Exception as inner:
        # Bahkan update ke DB pun gagal — log saja, jangan crash thread
        print(f"[{job_id[:8]}] ⚠️  Tidak bisa update status ke DB: {inner}")


# ─────────────────────────────────────────────
# Core inference — dipanggil oleh run_tif_upload_job
# ─────────────────────────────────────────────

def _run_inference(job_id: str, file_path: str, conf: float, batch_size: int) -> None:
    """
    Jalankan inference YOLO tile per tile.
    Setiap tahap dikungkung try/except tersendiri sehingga error apapun
    (model load gagal, rasterio crash, DB lost connection, OOM, dll.)
    langsung mengubah status menjadi failed — bukan exception yang menggantung.
    """
    # Import di sini agar error import (ultralytics belum install, dll)
    # terdeteksi sebagai job failure, bukan server crash
    try:
        from running_model3 import count_per_class, get_model
        from tile3 import generate_tiles_from_tiff, get_tif_info
    except ImportError as e:
        _fail_job(job_id, f"Dependency tidak terinstall: {e}")
        return

    start = time.monotonic()

    # ── 1. Baca metadata ─────────────────────────────────────────
    try:
        update_job(job_id, status="processing", progress=2,
                   message="Membaca metadata TIF...")
    except Exception as e:
        print(f"[{job_id[:8]}] ⚠️  DB update gagal di step 1: {e}")
        # Lanjut saja — DB mungkin sesaat tidak bisa dihubungi

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
            message=(
                f"TIF {tif_info['width']}×{tif_info['height']}px "
                f"({tif_info['file_size_mb']} MB) · "
                f"~{total_grid} grid tiles · Memuat model..."
            ),
        )
    except Exception as e:
        print(f"[{job_id[:8]}] ⚠️  DB update gagal setelah baca metadata: {e}")

    # ── 2. Load model ─────────────────────────────────────────────
    try:
        model       = get_model(MODEL_PATH)
        class_names = model.names
        accumulated = {name: 0 for name in class_names.values()}
    except Exception as e:
        _fail_job(job_id, f"Gagal memuat model YOLO ({MODEL_PATH}): {e}", e)
        return

    try:
        update_job(job_id, progress=5,
                   message=f"Model siap. Mulai inferensi ~{total_grid} tiles...")
    except Exception as e:
        print(f"[{job_id[:8]}] ⚠️  DB update gagal setelah model load: {e}")

    # ── 3. Inferensi tile per tile ────────────────────────────────
    tiles_done    = 0
    last_db_write = time.monotonic()
    speed_window: list[float] = []

    try:
        tile_gen = generate_tiles_from_tiff(file_path, batch_size=batch_size)
    except Exception as e:
        _fail_job(job_id, f"Gagal membuka TIF untuk tiling: {e}", e)
        return

    for batch, grid_idx in tile_gen:
        try:
            t0      = time.monotonic()
            results = model.predict(source=batch, conf=conf, verbose=False, max_det=2000)

            # Akumulasi count per class
            batch_counts = count_per_class(results, model)
            for cls in class_names.values():
                accumulated[cls] = accumulated.get(cls, 0) + batch_counts.get(cls, 0)

            tiles_done += len(batch)

            # Hitung kecepatan dengan rolling window 20 batch
            elapsed_per_tile = (time.monotonic() - t0) / max(len(batch), 1)
            speed_window.append(elapsed_per_tile)
            if len(speed_window) > 20:
                speed_window.pop(0)

            avg_spt       = sum(speed_window) / len(speed_window)
            tiles_per_sec = round(1 / avg_spt, 2) if avg_spt > 0 else 0
            progress      = min(5 + int((grid_idx / total_grid) * 94), 99) if total_grid else 99
            eta           = int((total_grid - grid_idx) * avg_spt) if avg_spt > 0 else None
            running_total = sum(accumulated.values())

            # Tulis ke DB maksimal 1x per detik — hindari hammering DB
            # Gunakan pg_notify untuk realtime update ke Supabase
            now = time.monotonic()
            if now - last_db_write >= 1.0:
                try:
                    update_job(
                        job_id,
                        progress=progress,
                        tiles_processed=tiles_done,
                        tiles_per_second=tiles_per_sec,
                        eta_seconds=eta,
                        message=(
                            f"Tile {tiles_done} selesai (grid {grid_idx}/{total_grid}) · "
                            f"{tiles_per_sec} tile/s · ETA {_fmt_duration(eta)} · "
                            f"deteksi sementara: {running_total}"
                        ),
                    )
                    # Notify Supabase Realtime listeners
                    notify_job_progress(job_id, {
                        "progress": progress,
                        "tiles_processed": tiles_done,
                        "total_grid_tiles": total_grid,
                        "tiles_per_second": tiles_per_sec,
                        "eta_seconds": eta,
                        "running_total": running_total,
                        "status": "processing",
                    })
                except Exception as db_err:
                    # DB sesaat tidak bisa dihubungi — log tapi lanjut inference
                    print(f"[{job_id[:8]}] ⚠️  DB write gagal (akan retry): {db_err}")
                last_db_write = now

        except Exception as e:
            # Error pada SATU batch tile — catat, tandai failed, hentikan inference
            tb = traceback.format_exc()
            print(f"[{job_id[:8]}] ❌ Error saat inference batch (grid={grid_idx}):\n{tb}")
            _fail_job(job_id, f"Error saat inference tile {grid_idx}: {e}", e)
            return
        finally:
            del batch
            if 'results' in dir():
                del results
            gc.collect()

    # ── 4. Finalisasi ─────────────────────────────────────────────
    elapsed     = round(time.monotonic() - start, 1)
    final_total = sum(accumulated.values())
    accumulated["total"] = final_total

    done_message = (
        f"Selesai dalam {_fmt_duration(int(elapsed))}. "
        f"Terdeteksi {final_total} pohon sawit dari {tiles_done} tile valid."
    )

    try:
        update_job(
            job_id,
            status="done",
            progress=100,
            total_detected=final_total,
            class_counts=accumulated,
            tiles_processed=tiles_done,
            tiles_per_second=0,
            eta_seconds=0,
            temp_file_path=None,
            message=done_message,
        )
        # Notify selesai ke Supabase Realtime
        notify_job_progress(job_id, {
            "progress": 100,
            "status": "done",
            "total_detected": final_total,
            "class_counts": accumulated,
            "tiles_processed": tiles_done,
            "message": done_message,
        })
    except Exception as e:
        print(f"[{job_id[:8]}] ⚠️  Gagal update final ke DB: {e}")
        # Hasil inferensi sudah selesai tapi tidak bisa disimpan — tetap tandai failed
        _fail_job(job_id, f"Inferensi selesai tapi gagal simpan hasil ke DB: {e}", e)
        return

    print(f"[{job_id[:8]}] ✅ Done — {final_total} pohon · {tiles_done} tiles · {elapsed}s")


# ─────────────────────────────────────────────
# Job runner — dipanggil oleh executor.submit()
# ─────────────────────────────────────────────

def run_tif_upload_job(
    job_id: str, file_path: str, conf: float, batch_size: int
) -> None:
    """
    Runner untuk file TIF yang sudah ada di disk (upload langsung).
    finally: file SELALU dihapus, tidak peduli sukses/gagal/exception.
    """
    print(f"[{job_id[:8]}] 🚀 Mulai inference — {file_path}")
    try:
        _run_inference(job_id, file_path, conf, batch_size)

    except Exception as e:
        # Catch-all: exception yang tidak tertangkap di _run_inference
        tb = traceback.format_exc()
        print(f"[{job_id[:8]}] ❌ Unhandled exception:\n{tb}")
        _fail_job(job_id, f"Error tidak terduga: {e}", e)

    finally:
        # File temp SELALU dihapus — sukses maupun gagal
        safe_delete(file_path, f"tif_upload job={job_id[:8]}")
        gc.collect()
        print(f"[{job_id[:8]}] 🔓 Worker slot dibebaskan")


# ─────────────────────────────────────────────
# Public API — dipanggil dari main.py
# ─────────────────────────────────────────────

def submit_tif_upload(job_id: str, file_path: str, conf: float, batch_size: int) -> None:
    """
    Submit job ke thread pool. Return langsung — tidak blocking.
    ThreadPoolExecutor otomatis antri kalau semua slot penuh.
    """
    _executor.submit(run_tif_upload_job, job_id, file_path, conf, batch_size)
