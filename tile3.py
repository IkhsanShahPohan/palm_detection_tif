import gc
import os

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


# def download_from_gdrive(url, output_path):
#     gdown.download(url, output_path, quiet=False, fuzzy=True)


def get_tif_info(input_tiff: str, tile_size: int = 640) -> dict:
    """
    Baca metadata TIF tanpa membaca piksel.
    Return info dimensi, jumlah grid tile (upper bound),
    dan ukuran file. Digunakan untuk estimasi progress sebelum
    inferensi dimulai.

    Catatan: total_grid_tiles adalah upper bound karena tile blank/invalid
    akan diskip saat inferensi, jadi actual_tiles <= total_grid_tiles.
    """
    with rasterio.open(input_tiff) as src:
        img_w, img_h = src.width, src.height
        crs = str(src.crs) if src.crs else "unknown"
        bands = src.count
        dtype = str(src.dtypes[0]) if src.dtypes else "unknown"

    full_cols = img_w // tile_size
    full_rows = img_h // tile_size
    total_grid_tiles = full_cols * full_rows

    file_size_bytes = os.path.getsize(input_tiff)

    return {
        "width": img_w,
        "height": img_h,
        "bands": bands,
        "dtype": dtype,
        "crs": crs,
        "tile_size": tile_size,
        "total_grid_tiles": total_grid_tiles,
        "full_cols": full_cols,
        "full_rows": full_rows,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / 1024 / 1024, 2),
    }


def generate_tiles_from_tiff(input_tiff, tile_size=640, min_content=0.05, batch_size=4, simulate_jpeg=True):
    """
    Generator: baca TIF per-tile, yield (batch, grid_index).
    grid_index = posisi tile terakhir dalam grid (1-based),
    dipakai untuk progress tracking yang akurat.
    """

    def is_valid_tile(arr):
        gray = arr.mean(axis=2)
        mask = (gray > 10) & (gray < 245)
        return mask.sum() / mask.size >= min_content

    with rasterio.open(input_tiff) as src:
        img_w, img_h = src.width, src.height

        sample_size = min(4096, img_w, img_h)
        cx = img_w // 2 - sample_size // 2
        cy = img_h // 2 - sample_size // 2
        sample_win = Window(cx, cy, sample_size, sample_size)
        sample_data = src.read([1, 2, 3], window=sample_win).astype(np.float32)

        global_p2, global_p98 = [], []
        for i in range(3):
            band = sample_data[i]
            valid = band[band > 0]
            if valid.size > 0:
                global_p2.append(float(np.percentile(valid, 2)))
                global_p98.append(float(np.percentile(valid, 98)))
            else:
                global_p2.append(0.0)
                global_p98.append(255.0)

        del sample_data
        gc.collect()

        def to_uint8_global(data):
            if data.dtype == np.uint8:
                return data
            result = np.zeros_like(data, dtype=np.uint8)
            for i in range(3):
                band = data[i].astype(np.float32)
                p2, p98 = global_p2[i], global_p98[i]
                if p98 <= p2:
                    result[i] = np.clip(band, 0, 255).astype(np.uint8)
                    continue
                band = np.clip(band, p2, p98)
                band = (band - p2) / (p98 - p2) * 255.0
                result[i] = band.astype(np.uint8)
            return result

        cols = list(range(0, img_w, tile_size))
        rows = list(range(0, img_h, tile_size))
        batch = []
        grid_index = 0

        for row_off in rows:
            for col_off in cols:
                win_w = min(tile_size, img_w - col_off)
                win_h = min(tile_size, img_h - row_off)

                grid_index += 1

                if win_w < tile_size or win_h < tile_size:
                    continue

                window = Window(col_off, row_off, win_w, win_h)
                data = src.read([1, 2, 3], window=window)

                if data.dtype == np.uint8:
                    arr = np.moveaxis(data, 0, -1)
                else:
                    arr = np.moveaxis(to_uint8_global(data), 0, -1)

                del data

                if not is_valid_tile(arr):
                    del arr
                    continue

                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                del arr

                if simulate_jpeg:
                    _, encoded = cv2.imencode('.jpg', arr_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    arr_final = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    del encoded
                else:
                    arr_final = arr_bgr

                del arr_bgr
                batch.append(arr_final)

                if len(batch) == batch_size:
                    yield batch, grid_index
                    del batch
                    gc.collect()
                    batch = []

        if batch:
            yield batch, grid_index
            del batch
            gc.collect()
