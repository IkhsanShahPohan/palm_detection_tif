"""
test_api.py  ·  v6
Endpoint yang diuji:
  POST  /detect/tif
  GET   /jobs          ← ganti /history + /history/me
  GET   /jobs/{job_id} ← ganti /status/{job_id}
  GET   /health
  GET   /admin/temp/list
  DELETE /admin/temp/cleanup
"""

import io
import json
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock

# ── Env sebelum import apapun ─────────────────
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
os.environ["API_KEY"]      = "test-key-rahasia"
os.environ["MODEL_PATH"]   = "best_v3.pt"
os.environ["TEMP_DIR"]     = "temp_test_uploads"

# ── Mock modul eksternal ──────────────────────
MOCKS = {
    "psycopg2": MagicMock(), "psycopg2.pool": MagicMock(),
    "psycopg2.extras": MagicMock(), "ultralytics": MagicMock(),
    "rasterio": MagicMock(), "cv2": MagicMock(),
    "gdown": MagicMock(), "running_model3": MagicMock(),
    "tile3": MagicMock(), "orjson": MagicMock(),
}
for mod, mock in MOCKS.items():
    sys.modules[mod] = mock

class _FakeAioFile:
    async def write(self, data): pass
    async def close(self): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

aiofiles_mock = MagicMock()
aiofiles_mock.open = MagicMock(return_value=_FakeAioFile())
sys.modules["aiofiles"] = aiofiles_mock

from fastapi.responses import JSONResponse
import fastapi.responses as _fr
_fr.ORJSONResponse = JSONResponse

db_mock     = MagicMock()
worker_mock = MagicMock()
worker_mock.MAX_CONCURRENT = 2
worker_mock.safe_delete    = MagicMock()
worker_mock._executor      = MagicMock()

db_mock.init_db           = MagicMock(return_value=None)
db_mock.get_stale_jobs    = MagicMock(return_value=[])
db_mock.count_active_jobs = MagicMock(return_value=0)
db_mock.create_job        = MagicMock(return_value=None)
db_mock.update_job        = MagicMock(return_value=None)
db_mock.get_job           = MagicMock(return_value=None)
db_mock.get_history       = MagicMock(return_value={
    "total": 0, "limit": 20, "offset": 0, "data": []
})

sys.modules["database"] = db_mock
sys.modules["worker"]   = worker_mock

import importlib
import main as main_module
importlib.reload(main_module)

from fastapi.testclient import TestClient
client = TestClient(main_module.app, raise_server_exceptions=False)

H      = {"X-API-Key": "test-key-rahasia", "X-User-ID": "user_123"}
H_NOID = {"X-API-Key": "test-key-rahasia"}
H_BKEY = {"X-API-Key": "key-salah", "X-User-ID": "user_123"}


def make_job(status="queued", total=0, user="user_123"):
    counts = {}
    if total:
        counts = {"palm": total, "fade_palm": 0, "total": total}
    return {
        "id":               "job-abc-123",
        "user_id":          user,
        "status":           status,
        "progress":         100 if status == "done" else 0,
        "class_counts":     counts,
        "tiles_processed":  50  if status == "done" else 0,
        "total_grid_tiles": 55,
        "eta_seconds":      None,
        "elapsed_seconds":  268 if status == "done" else None,
        "tif_info":         {"width": 5000, "height": 4000},
        "filename":         "kebun.tif",
        "file_size_bytes":  84410000,
        "created_at":       datetime.now().isoformat(),
        "updated_at":       datetime.now().isoformat(),
    }


# ═════════════════════════════════════════════
# TEST: Autentikasi
# ═════════════════════════════════════════════
class TestAuth(unittest.TestCase):

    def test_tanpa_api_key_401_atau_422(self):
        r = client.get("/jobs", headers={"X-User-ID": "u1"})
        self.assertIn(r.status_code, [401, 422])

    def test_api_key_salah_401(self):
        r = client.get("/jobs", headers=H_BKEY)
        self.assertEqual(r.status_code, 401)

    def test_api_key_benar_200(self):
        r = client.get("/jobs", headers=H)
        self.assertEqual(r.status_code, 200)

    def test_health_tanpa_auth(self):
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

    def test_file_kosong_400(self):
        r = client.post("/detect/tif", headers=H,
            files={"file": ("kebun.tif", io.BytesIO(b""), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 400)

    def test_quota_penuh_429(self):
        db_mock.count_active_jobs.return_value = 3
        r = client.post("/detect/tif", headers=H,
            files={"file": ("k.tif", io.BytesIO(b"data"), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 429)

    def test_upload_sukses_response(self):
        r = client.post("/detect/tif", headers=H,
            files={"file": ("kebun.tif", io.BytesIO(b"data" * 200), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for f in ["id", "status", "user_id", "filename", "size_bytes"]:
            self.assertIn(f, body, f"Field '{f}' tidak ada")
        for f in ["poll_url", "realtime_channel", "realtime_event", "message", "size_mb"]:
            self.assertNotIn(f, body, f"Field '{f}' seharusnya tidak ada")
        self.assertEqual(body["status"],  "queued")
        self.assertEqual(body["user_id"], "user_123")
        self.assertGreater(body["size_bytes"], 0)
        self.assertEqual(worker_mock.submit_tif_upload.call_count, 1)
        print(f"\n  ✓ upload response bersih: {list(body.keys())}")

    def test_user_id_anonymous_jika_tidak_ada_header(self):
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

    def test_endpoint_lama_tidak_ada(self):
        r1 = client.get("/status/abc", headers=H)
        r2 = client.get("/history",    headers=H)
        r3 = client.get("/history/me", headers=H)
        self.assertEqual(r1.status_code, 404, "/status/{id} seharusnya 404")
        self.assertEqual(r2.status_code, 404, "/history seharusnya 404")
        self.assertEqual(r3.status_code, 404, "/history/me seharusnya 404")
        print(f"\n  ✓ Endpoint lama /status, /history, /history/me sudah tidak ada")


# ═════════════════════════════════════════════
# TEST: GET /jobs/{job_id}
# ═════════════════════════════════════════════
class TestGetJob(unittest.TestCase):

    def test_job_tidak_ada_404(self):
        db_mock.get_job.return_value = None
        r = client.get("/jobs/tidak-ada", headers=H)
        self.assertEqual(r.status_code, 404)

    def test_job_ada_struktur_benar(self):
        db_mock.get_job.return_value = make_job("processing")
        r = client.get("/jobs/job-abc-123", headers=H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for f in ["id", "user_id", "status", "progress",
                  "class_counts", "tiles_processed", "total_grid_tiles",
                  "eta_seconds", "elapsed_seconds", "tif_info",
                  "filename", "created_at", "updated_at"]:
            self.assertIn(f, body, f"Field '{f}' tidak ada")
        for f in ["total_detected", "message", "tiles_per_second",
                  "file_size_bytes", "source_type"]:
            self.assertNotIn(f, body, f"Field lama '{f}' masih ada")

    def test_job_done_elapsed_seconds(self):
        db_mock.get_job.return_value = make_job("done", total=2739)
        r = client.get("/jobs/job-abc-123", headers=H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"],  "done")
        self.assertEqual(body["progress"], 100)
        self.assertIsNotNone(body["elapsed_seconds"])
        self.assertEqual(body["class_counts"]["total"], 2739)
        print(f"\n  ✓ elapsed_seconds={body['elapsed_seconds']}s, "
              f"class_counts.total={body['class_counts']['total']}")

    def test_job_user_lain_403(self):
        db_mock.get_job.return_value = make_job("done", user="user_lain")
        r = client.get("/jobs/job-abc-123", headers=H)
        self.assertEqual(r.status_code, 403)
        print(f"\n  ✓ akses job user lain ditolak 403")


# ═════════════════════════════════════════════
# TEST: GET /jobs
# ═════════════════════════════════════════════
class TestListJobs(unittest.TestCase):

    def _mock(self, jobs, total=None):
        db_mock.get_history.return_value = {
            "total":  total if total is not None else len(jobs),
            "limit":  20, "offset": 0, "data": jobs,
        }

    def test_struktur_response(self):
        self._mock([make_job("done")])
        r = client.get("/jobs", headers=H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for f in ["user_id", "total", "limit", "offset", "has_more", "data"]:
            self.assertIn(f, body)
        self.assertEqual(body["user_id"], "user_123")

    def test_user_id_dari_header(self):
        self._mock([])
        client.get("/jobs", headers=H)
        call_kwargs = db_mock.get_history.call_args.kwargs
        self.assertEqual(call_kwargs.get("user_id"), "user_123")

    def test_user_id_tidak_bisa_override_via_query(self):
        self._mock([])
        client.get("/jobs?user_id=user_lain", headers=H)
        call_kwargs = db_mock.get_history.call_args.kwargs
        self.assertNotEqual(call_kwargs.get("user_id"), "user_lain")
        print(f"\n  ✓ user_id tidak bisa di-override via query param")

    def test_filter_status(self):
        self._mock([])
        client.get("/jobs?status=done", headers=H)
        self.assertEqual(db_mock.get_history.call_args.kwargs.get("status"), "done")

    def test_filter_date(self):
        self._mock([])
        client.get("/jobs?date_from=2024-01-01&date_to=2024-12-31", headers=H)
        kw = db_mock.get_history.call_args.kwargs
        self.assertEqual(kw.get("date_from"), "2024-01-01")
        self.assertEqual(kw.get("date_to"),   "2024-12-31")

    def test_has_more_true(self):
        self._mock([make_job()] * 20, total=50)
        self.assertTrue(client.get("/jobs", headers=H).json()["has_more"])

    def test_has_more_false(self):
        self._mock([make_job()])
        self.assertFalse(client.get("/jobs", headers=H).json()["has_more"])

    def test_kosong_valid(self):
        self._mock([])
        body = client.get("/jobs", headers=H).json()
        self.assertEqual(body["total"], 0)
        self.assertEqual(body["data"],  [])

    def test_limit_max_100(self):
        self._mock([])
        r = client.get("/jobs?limit=200", headers=H)
        self.assertIn(r.status_code, [200, 422])


# ═════════════════════════════════════════════
# TEST: Admin & Health
# ═════════════════════════════════════════════
class TestAdminHealth(unittest.TestCase):

    def test_health_struktur(self):
        r = client.get("/health")
        self.assertIn(r.status_code, [200, 500])
        if r.status_code == 200:
            for f in ["status", "database", "max_workers", "version"]:
                self.assertIn(f, r.json())

    def test_temp_list(self):
        r = client.get("/admin/temp/list", headers=H)
        self.assertEqual(r.status_code, 200)
        for f in ["dir", "count", "total_mb", "files"]:
            self.assertIn(f, r.json())

    def test_temp_cleanup(self):
        r = client.delete("/admin/temp/cleanup", headers=H)
        self.assertEqual(r.status_code, 200)
        for f in ["removed", "errors", "count_removed"]:
            self.assertIn(f, r.json())


# ═════════════════════════════════════════════
# TEST: Concurrent & isolasi
# ═════════════════════════════════════════════
class TestIsolasi(unittest.TestCase):

    def test_multi_user_submit_bersamaan(self):
        db_mock.count_active_jobs.return_value = 0
        worker_mock.submit_tif_upload.reset_mock()
        for uid in ["user_A", "user_B", "user_C"]:
            r = client.post("/detect/tif",
                headers={"X-API-Key": "test-key-rahasia", "X-User-ID": uid},
                files={"file": ("k.tif", io.BytesIO(b"x" * 100), "image/tiff")},
                data={"conf": "0.5", "batch_size": "4"})
            self.assertEqual(r.status_code, 200, f"user {uid} gagal submit")
        self.assertEqual(worker_mock.submit_tif_upload.call_count, 3)
        print(f"\n  ✓ 3 user berbeda submit bersamaan: semua queued")

    def test_response_tidak_ada_field_lebih(self):
        db_mock.count_active_jobs.return_value = 0
        worker_mock.submit_tif_upload.reset_mock()
        r = client.post("/detect/tif", headers=H,
            files={"file": ("k.tif", io.BytesIO(b"x" * 100), "image/tiff")},
            data={"conf": "0.5", "batch_size": "4"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for f in ["poll_url", "realtime_channel", "realtime_event", "message", "size_mb"]:
            self.assertNotIn(f, body, f"Field '{f}' seharusnya tidak ada")
        print(f"\n  ✓ response bersih, tidak ada field berlebih")


# ═════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("temp_test_uploads", exist_ok=True)
    print("=" * 60)
    print("Palm Tree Detection API v6 — Test Suite")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestAuth, TestUploadTIF, TestGetJob,
                TestListJobs, TestAdminHealth, TestIsolasi]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print()
    print("=" * 60)
    if result.wasSuccessful():
        print(f"✅  Semua {result.testsRun} test LULUS")
    else:
        print(f"❌  {len(result.failures)} gagal · {len(result.errors)} error "
              f"dari {result.testsRun} test")
        for f in result.failures + result.errors:
            print(f"\n  {'FAIL' if f in result.failures else 'ERROR'}: {f[0]}\n  {f[1]}")
    print("=" * 60)

    import sys as _sys
    _sys.exit(0 if result.wasSuccessful() else 1)
