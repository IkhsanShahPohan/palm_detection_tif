from ultralytics import YOLO

_model_cache = {}


def get_model(model_path: str) -> YOLO:
    if model_path not in _model_cache:
        print(f"📦 Loading model: {model_path}")
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


def count_per_class(results: list, model: YOLO) -> dict:
    """
    Hitung jumlah deteksi per class dari hasil prediksi.
    Mengembalikan dict: { class_name: count, ..., "total": total_count }
    """
    class_names = model.names  # {0: 'class_a', 1: 'class_b', 2: 'class_c'}
    counts = {name: 0 for name in class_names.values()}

    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for cls_id in r.boxes.cls.tolist():
                cls_name = class_names.get(int(cls_id), f"class_{int(cls_id)}")
                counts[cls_name] = counts.get(cls_name, 0) + 1

    counts["total"] = sum(counts.values())
    return counts


def merge_class_counts(a: dict, b: dict) -> dict:
    """
    Gabungkan dua dict class count.
    """
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result


def run_detection_stream(model_path: str, tile_generator, conf: float = 0.5):
    """Fungsi lama — dipertahankan untuk kompatibilitas."""
    model = get_model(model_path)
    total = 0
    for batch in tile_generator:
        results = model.predict(source=batch, conf=conf, verbose=False)
        for r in results:
            total += len(r.boxes)
    return total
