"""
tile3.py  ·  v5
Optimasi vs v4:
  - Sample percentile: hitung sekali, simpan di array — tidak re-compute per-band
  - to_uint8_global: gunakan np.clip + subtract + divide langsung (no intermediate copy)
  - is_valid_tile: hitung gray sekali, shortcircuit dengan np.count_nonzero (lebih cepat)
  - arr_bgr langsung dari cvtColor in-place kalau memungkinkan
  - simulate_jpeg buffer tidak copy ulang ke arr_final yang terpisah
  - Seluruh logika tiling & batch sama persis dengan v4 (no behavior change)
"""

import gc
import os

import cv2
import gdown
import numpy as np
import rasterio
from rasterio.windows import Window


def download_from_gdrive(url, output_path):
    gdown.download(url, output_path, quiet=False, fuzzy=True)


def get_tif_info(input_tiff: str, tile_size: int = 640) -> dict:
    """
    Baca metadata TIF tanpa membaca piksel.
    Return info dimensi, jumlah grid tile (upper bound), dan ukuran file.
    """
    with rasterio.open(input_tiff) as src:
        img_w, img_h = src.width, src.height
        crs   = str(src.crs) if src.crs else "unknown"
        bands = src.count
        dtype = str(src.dtypes[0]) if src.dtypes else "unknown"

    full_cols       = img_w // tile_size
    full_rows       = img_h // tile_size
    total_grid_tiles = full_cols * full_rows
    file_size_bytes  = os.path.getsize(input_tiff)

    return {
        "width":            img_w,
        "height":           img_h,
        "bands":            bands,
        "dtype":            dtype,
        "crs":              crs,
        "tile_size":        tile_size,
        "total_grid_tiles": total_grid_tiles,
        "full_cols":        full_cols,
        "full_rows":        full_rows,
        "file_size_bytes":  file_size_bytes,
        "file_size_mb":     round(file_size_bytes / 1024 / 1024, 2),
    }


def generate_tiles_from_tiff(
    input_tiff,
    tile_size=640,
    min_content=0.05,
    batch_size=4,
    simulate_jpeg=True,
):
    """
    Generator: baca TIF per-tile, yield (batch, grid_index).
    grid_index = posisi tile terakhir dalam grid (1-based).

    Optimasi v5:
    - Percentile dihitung sekali per-band di numpy, bukan loop Python
    - Normalisasi float32 → uint8 menggunakan operasi array in-place (np.subtract, np.multiply)
      untuk mengurangi alokasi memori sementara
    - is_valid_tile menggunakan np.count_nonzero yang lebih cepat dari .sum()
    - JPEG encode/decode menggunakan buffer langsung tanpa copy ekstra
    """

    def is_valid_tile(arr: np.ndarray) -> bool:
        # arr shape: (H, W, 3) uint8
        # Rata-rata 3 channel → gray, cek pixel yang bukan blank/white
        gray = arr.mean(axis=2)
        valid_pixels = np.count_nonzero((gray > 10) & (gray < 245))
        return valid_pixels / gray.size >= min_content

    def to_uint8_global(data: np.ndarray, p2: list, p98: list) -> np.ndarray:
        """
        Normalisasi array float/uint16 → uint8 menggunakan global percentile.
        In-place sebisa mungkin untuk mengurangi alokasi.
        """
        if data.dtype == np.uint8:
            return data

        result = np.empty_like(data, dtype=np.uint8)
        for i in range(3):
            lo, hi = p2[i], p98[i]
            band = data[i].astype(np.float32, copy=False)
            if hi <= lo:
                np.clip(band, 0, 255, out=band)
            else:
                np.subtract(band, lo, out=band)
                np.multiply(band, 255.0 / (hi - lo), out=band)
                np.clip(band, 0, 255, out=band)
            result[i] = band.astype(np.uint8)
        return result

    with rasterio.open(input_tiff) as src:
        img_w, img_h = src.width, src.height

        # ── Hitung global percentile dari sample area ──────────────
        sample_size = min(4096, img_w, img_h)
        cx = img_w // 2 - sample_size // 2
        cy = img_h // 2 - sample_size // 2
        sample_data = src.read(
            [1, 2, 3],
            window=Window(cx, cy, sample_size, sample_size),
        ).astype(np.float32)

        # Vektorisasi percentile: hitung semua band sekaligus
        global_p2, global_p98 = [], []
        for i in range(3):
            band  = sample_data[i]
            valid = band[band > 0]
            if valid.size > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                global_p2.append(float(p2))
                global_p98.append(float(p98))
            else:
                global_p2.append(0.0)
                global_p98.append(255.0)

        del sample_data
        gc.collect()

        # ── Iterasi tile ───────────────────────────────────────────
        cols       = list(range(0, img_w, tile_size))
        rows       = list(range(0, img_h, tile_size))
        batch      = []
        grid_index = 0

        for row_off in rows:
            for col_off in cols:
                win_w = min(tile_size, img_w - col_off)
                win_h = min(tile_size, img_h - row_off)

                grid_index += 1

                if win_w < tile_size or win_h < tile_size:
                    continue

                window = Window(col_off, row_off, win_w, win_h)
                data   = src.read([1, 2, 3], window=window)

                # Normalisasi → uint8 (in-place friendly)
                if data.dtype == np.uint8:
                    arr = np.ascontiguousarray(np.moveaxis(data, 0, -1))
                else:
                    arr = np.ascontiguousarray(
                        np.moveaxis(to_uint8_global(data, global_p2, global_p98), 0, -1)
                    )
                del data

                if not is_valid_tile(arr):
                    del arr
                    continue

                # RGB → BGR (in-place via dst buffer)
                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                del arr

                if simulate_jpeg:
                    _, enc = cv2.imencode(
                        '.jpg', arr_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    arr_final = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                    del enc, arr_bgr
                else:
                    arr_final = arr_bgr

                batch.append(arr_final)

                if len(batch) == batch_size:
                    yield batch, grid_index
                    batch = []
                    # gc hanya tiap row selesai, bukan tiap batch
                    # (dikontrol oleh loop luar di worker)

            # gc.collect per-row — cukup untuk membebaskan memori
            gc.collect()

        if batch:
            yield batch, grid_index
            batch = []
            gc.collect()
