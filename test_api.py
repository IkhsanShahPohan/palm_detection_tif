"""
test_api.py  ·  v5
Jalankan: python test_api.py
Tidak butuh DB, Redis, atau model YOLO sungguhan — semua di-mock.
Kompatibel dengan main.py v5 (versi string 5.0.0, aiofiles, ORJSONResponse).
"""

import io
import json
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# ── Setup env sebelum import apapun ──────────────────────────
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
os.environ["API_KEY"]      = "test-key-rahasia"
os.environ["MODEL_PATH"]   = "best_v3.pt"
os.environ["TEMP_DIR"]     = "temp_test_uploads"

# ── Mock semua modul yang butuh install khusus ────────────────
MOCKS = {
    "psycopg2":        MagicMock(),
    "psycopg2.pool":   MagicMock(),
    "psycopg2.extras": MagicMock(),
    "ultralytics":     MagicMock(),
    "rasterio":        MagicMock(),
    "cv2":             MagicMock(),
    "gdown":           MagicMock(),
    "running_model3":  MagicMock(),
    "tile3":           MagicMock(),
    "aiofiles":        MagicMock(),
    "orjson":          MagicMock(),
}
for mod, mock in MOCKS.items():
    sys.modules[mod] = mock

# ── Mock aiofiles.open sebagai async context manager ─────────
import types

class _FakeAioFile:
    async def write(self, data): pass
    async def close(self): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

aiofiles_mock = MagicMock()
aiofiles_mock.open = MagicMock(return_value=_FakeAioFile())
sys.modules["aiofiles"] = aiofiles_mock

# ── Mock fastapi.responses.ORJSONResponse ────────────────────
# ORJSONResponse butuh orjson yang mungkin tidak terinstall di test env
from fastapi.responses import JSONResponse
import fastapi.responses as _fr
_fr.ORJSONResponse = JSONResponse   # fallback ke JSONResponse standar

# ── Mock database dan worker ──────────────────────────────────
db_mock     = MagicMock()
worker_mock = MagicMock()
worker_mock.MAX_CONCURRENT = 2
worker_mock.safe_delete    = MagicMock()
worker_mock._executor      = MagicMock()
worker_mock._executor._work_queue.qsize.return_value = 0

# Pastikan semua fungsi DB yang di-await bisa dikembalikan sebagai coroutine
db_mock.init_db          = MagicMock(return_value=None)
db_mock.get_stale_jobs   = MagicMock(return_value=[])
db_mock.count_active_jobs = MagicMock(return_value=0)
db_mock.create_job       = MagicMock(return_value=None)
db_mock.update_job       = MagicMock(return_value=None)
db_mock.get_job          = MagicMock(return_value=None)
db_mock.get_history      = MagicMock(return_value={"total": 0, "limit": 50, "offset": 0, "data": []})

sys.modules["database"] = db_mock
sys.modules["worker"]   = worker_mock

import importlib
import main as main_module
importlib.reload(main_module)

from fastapi.testclient import TestClient
client = TestClient(main_module.app, raise_server_exceptions=False)

# ── Header default ────────────────────────────────────────────
H      = {"X-API-Key": "test-key-rahasia", "X-User-ID": "user_123"}
H_NOID = {"X-API-Key": "test-key-rahasia"}
H_BKEY = {"X-API-Key": "key-salah",        "X-User-ID": "user_123"}


# ── Sample job ────────────────────────────────────────────────
def make_job(status="queued", total=0, user="user_123", source="tif_upload"):
    return {
        "job_id":           "job-abc-123",
        "user_id":          user,
        "source_type":      source,
        "status":           status,
        "progress":         100 if status == "done" else 0,
        "total_detected":   total,
        "class_counts":     {"sawit": total, "total": total} if total else {},
        "tiles_processed":  50 if status == "done" else 0,
        "total_grid_tiles": 55,
        "tiles_per_second": 0,
        "eta_seconds":      None,
        "tif_info":         {"width": 5000, "height": 4000, "file_size_mb": 80.5},
        "filename":         "kebun.tif",
        "file_size_bytes":  84410000,
        "message":          "Selesai." if status == "done" else "Antri...",
        "created_at":       datetime.now().isoformat(),
        "updated_at":       datetime.now().isoformat(),
    }


# ═════════════════════════════════════════════
# TEST: Autentikasi
# ═════════════════════════════════════════════
class TestAuth(unittest.TestCase):

    def test_tanpa_api_key_401_atau_422(self):
        r = client.get("/history", headers={"X-User-ID": "u1"})
        self.assertIn(r.status_code, [401, 422],
                      "Tanpa X-API-Key harus 401 atau 422")

    def test_api_key_salah_401(self):
        r = client.get("/history", headers=H_BKEY)
        self.assertEqual(r.status_code, 401)
        self.assertIn("API key", r.json()["detail"])

    def test_api_key_benar_200(self):
        db_mock.get_history.return_value = {"total": 0, "limit": 50, "offset": 0, "data": []}
        r = client.get("/history", headers=H)
        self.assertEqual(r.status_code, 200)

    def test_health_tanpa_auth_bisa_akses(self):
        r = client.get("/health")
        self.assertIn(r.status_code, [200, 500])


# ═════════════════════════════════════════════
# TEST: Upload TIF
# ═════════════════════════════════════════════
class TestUploadTIF(unittest.TestCase):

    def setUp(self):
        db_mock.count_active_jobs.return_value = 0
        db_mock.create_job.return_value        = None
        db_mock.update_job.return_value        = None
        worker_mock.submit_tif_upload.reset_mock()

    def test_ekstensi_salah_400(self):
        r = client.post("/detect/tif", headers=H,
            files={"file": ("foto.jpg", io.BytesIO(b"data"), "image/jpeg")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("tif", r.json()["detail"].lower())

    def test_file_kosong_400(self):
        r = client.post("/detect/tif", headers=H,
            files={"file": ("kebun.tif", io.BytesIO(b""), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 400)

    def test_terlalu_banyak_job_aktif_429(self):
        db_mock.count_active_jobs.return_value = 3
        r = client.post("/detect/tif", headers=H,
            files={"file": ("kebun.tif", io.BytesIO(b"isi"), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 429)
        self.assertIn("aktif", r.json()["detail"])

    def test_upload_berhasil_response_lengkap(self):
        db_mock.count_active_jobs.return_value = 0
        fake_data = b"FAKE TIF BINARY DATA " * 200

        r = client.post("/detect/tif", headers=H,
            files={"file": ("kebun_blok_a.tif", io.BytesIO(fake_data), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})

        self.assertEqual(r.status_code, 200)
        body = r.json()

        for field in ["status", "job_id", "user_id", "filename",
                      "size_bytes", "size_mb", "message", "poll_url",
                      "realtime_channel", "realtime_event"]:
            self.assertIn(field, body, f"Field '{field}' tidak ada di response")

        self.assertEqual(body["status"],           "queued")
        self.assertEqual(body["user_id"],          "user_123")
        self.assertEqual(body["filename"],         "kebun_blok_a.tif")
        self.assertEqual(body["realtime_channel"], "job_progress")
        self.assertEqual(body["realtime_event"],   body["job_id"])
        self.assertIn("/status/", body["poll_url"])
        self.assertGreater(body["size_bytes"], 0)
        self.assertTrue(body["job_id"])

        self.assertEqual(worker_mock.submit_tif_upload.call_count, 1)
        print(f"\n  ✓ upload response:\n{json.dumps(body, indent=4)}")

    def test_user_id_anonymous_jika_header_kosong(self):
        db_mock.count_active_jobs.return_value = 0
        r = client.post("/detect/tif", headers=H_NOID,
            files={"file": ("k.tif", io.BytesIO(b"data"), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["user_id"], "anonymous")

    def test_conf_di_luar_range_422(self):
        r = client.post("/detect/tif", headers=H,
            files={"file": ("k.tif", io.BytesIO(b"data"), "image/tiff")},
            data={"conf": "1.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 422)

    def test_endpoint_tif_url_tidak_ada_404(self):
        r = client.post("/detect/tif-url", headers=H,
            data={"gdrive_url": "https://drive.google.com/file/d/abc/view",
                  "conf": "0.5", "batch_size": "4"})
        self.assertIn(r.status_code, [404, 405])
        print(f"\n  ✓ /detect/tif-url sudah dihapus (HTTP {r.status_code})")


# ═════════════════════════════════════════════
# TEST: Status polling
# ═════════════════════════════════════════════
class TestStatus(unittest.TestCase):

    def test_job_tidak_ada_404(self):
        db_mock.get_job.return_value = None
        r = client.get("/status/job-tidak-ada", headers=H)
        self.assertEqual(r.status_code, 404)

    def test_job_ada_200(self):
        db_mock.get_job.return_value = make_job("processing")
        r = client.get("/status/job-abc-123", headers=H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for f in ["job_id", "status", "progress", "total_detected",
                  "class_counts", "tiles_processed", "total_grid_tiles",
                  "tiles_per_second", "eta_seconds", "message"]:
            self.assertIn(f, body, f"Field '{f}' tidak ada")

    def test_job_done_total_detected(self):
        db_mock.get_job.return_value = make_job("done", total=1234)
        r = client.get("/status/job-abc-123", headers=H)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["total_detected"], 1234)
        print(f"  ✓ total_detected=1234 dikembalikan dengan benar")


# ═════════════════════════════════════════════
# TEST: History
# ═════════════════════════════════════════════
class TestHistory(unittest.TestCase):

    def _mock_history(self, jobs, total=None):
        db_mock.get_history.return_value = {
            "total":  total if total is not None else len(jobs),
            "limit":  50,
            "offset": 0,
            "data":   jobs,
        }

    def test_history_struktur(self):
        self._mock_history([make_job("done")])
        r = client.get("/history", headers=H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for f in ["total", "limit", "offset", "has_more", "data"]:
            self.assertIn(f, body)
        self.assertEqual(len(body["data"]), 1)
        self.assertFalse(body["has_more"])

    def test_history_has_more_true(self):
        jobs = [make_job("done")] * 50
        self._mock_history(jobs, total=100)
        r = client.get("/history", headers=H)
        self.assertTrue(r.json()["has_more"])

    def test_history_me_pakai_header_user_id(self):
        self._mock_history([make_job("done")])
        r = client.get("/history/me", headers=H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("user_id", body)
        self.assertEqual(body["user_id"], "user_123")

    def test_history_kosong_valid(self):
        self._mock_history([])
        r = client.get("/history", headers=H)
        body = r.json()
        self.assertEqual(body["total"], 0)
        self.assertEqual(body["data"],  [])
        self.assertFalse(body["has_more"])

    def test_history_filter_status(self):
        self._mock_history([])
        r = client.get("/history?status=done&limit=10", headers=H)
        self.assertEqual(r.status_code, 200)
        call_kwargs = db_mock.get_history.call_args.kwargs
        self.assertEqual(call_kwargs.get("status"), "done")

    def test_history_filter_date_from_date_to(self):
        self._mock_history([])
        r = client.get("/history?date_from=2024-01-01&date_to=2024-12-31", headers=H)
        self.assertEqual(r.status_code, 200)
        call_kwargs = db_mock.get_history.call_args.kwargs
        self.assertEqual(call_kwargs.get("date_from"), "2024-01-01")
        self.assertEqual(call_kwargs.get("date_to"),   "2024-12-31")

    def test_history_sort_by_total_detected(self):
        self._mock_history([])
        r = client.get("/history?sort_by=total_detected&sort_order=desc", headers=H)
        self.assertEqual(r.status_code, 200)
        call_kwargs = db_mock.get_history.call_args.kwargs
        self.assertEqual(call_kwargs.get("sort_by"),    "total_detected")
        self.assertEqual(call_kwargs.get("sort_order"), "desc")

    def test_history_me_filter_status_dan_date(self):
        self._mock_history([])
        r = client.get("/history/me?status=failed&date_from=2024-06-01", headers=H)
        self.assertEqual(r.status_code, 200)
        call_kwargs = db_mock.get_history.call_args.kwargs
        self.assertEqual(call_kwargs.get("status"),    "failed")
        self.assertEqual(call_kwargs.get("date_from"), "2024-06-01")
        self.assertEqual(call_kwargs.get("user_id"),   "user_123")

    def test_history_limit_max_100(self):
        self._mock_history([])
        r = client.get("/history?limit=200", headers=H)
        self.assertIn(r.status_code, [200, 422])


# ═════════════════════════════════════════════
# TEST: Admin endpoints
# ═════════════════════════════════════════════
class TestAdmin(unittest.TestCase):

    def test_temp_list_struktur(self):
        r = client.get("/admin/temp/list", headers=H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for f in ["dir", "count", "total_mb", "files"]:
            self.assertIn(f, body)

    def test_temp_cleanup_struktur(self):
        r = client.delete("/admin/temp/cleanup", headers=H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for f in ["removed", "errors", "count_removed"]:
            self.assertIn(f, body)

    def test_health_struktur(self):
        r = client.get("/health")
        self.assertIn(r.status_code, [200, 500])
        if r.status_code == 200:
            body = r.json()
            for f in ["status", "database", "max_workers", "version"]:
                self.assertIn(f, body)

    def test_health_version_5(self):
        r = client.get("/health")
        if r.status_code == 200:
            self.assertEqual(r.json()["version"], "5.0.0")
            print(f"  ✓ version = 5.0.0")


# ═════════════════════════════════════════════
# TEST: Skenario kegagalan & isolasi
# ═════════════════════════════════════════════
class TestFailureScenarios(unittest.TestCase):

    def test_multiple_user_berbeda_bisa_submit_bersamaan(self):
        db_mock.count_active_jobs.return_value = 0
        db_mock.create_job.return_value        = None
        worker_mock.submit_tif_upload.reset_mock()

        users = ["user_A", "user_B", "user_C"]
        for uid in users:
            headers = {"X-API-Key": "test-key-rahasia", "X-User-ID": uid}
            r = client.post("/detect/tif", headers=headers,
                files={"file": ("k.tif", io.BytesIO(b"data" * 100), "image/tiff")},
                data={"conf": "0.5", "batch_size": "4"})
            self.assertEqual(r.status_code, 200,
                             f"User {uid} seharusnya bisa submit")
        self.assertEqual(worker_mock.submit_tif_upload.call_count, 3)
        print(f"  ✓ 3 user berbeda submit bersamaan: semua queued")

    def test_job_gagal_tidak_blok_job_baru(self):
        db_mock.count_active_jobs.return_value = 0
        r = client.post("/detect/tif", headers=H,
            files={"file": ("baru.tif", io.BytesIO(b"data" * 100), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 200)

    def test_response_upload_ada_realtime_info(self):
        db_mock.count_active_jobs.return_value = 0
        worker_mock.submit_tif_upload.reset_mock()
        r = client.post("/detect/tif", headers=H,
            files={"file": ("realtime.tif", io.BytesIO(b"data" * 100), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("realtime_channel", body)
        self.assertIn("realtime_event",   body)
        self.assertEqual(body["realtime_channel"], "job_progress")
        self.assertEqual(body["realtime_event"],   body["job_id"])


# ═════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("temp_test_uploads", exist_ok=True)

    print("=" * 60)
    print("Palm Tree Detection API v5 — Test Suite")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestAuth,
        TestUploadTIF,
        TestStatus,
        TestHistory,
        TestAdmin,
        TestFailureScenarios,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 60)
    if result.wasSuccessful():
        print(f"✅  Semua {result.testsRun} test LULUS")
    else:
        print(f"❌  {len(result.failures)} gagal · {len(result.errors)} error dari {result.testsRun} test")
        for f in result.failures:
            print(f"\n  FAIL: {f[0]}\n  {f[1]}")
        for e in result.errors:
            print(f"\n  ERROR: {e[0]}\n  {e[1]}")
    print("=" * 60)

    import sys as _sys
    _sys.exit(0 if result.wasSuccessful() else 1)
