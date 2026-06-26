import os
import math
import json
import time
import shutil
import importlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from contextlib import nullcontext

import h5py
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from src.metrics.eval_script import (
    calculate_metrics_from_predictions,
    get_match_sequence_plane_symmetry,
)

# ============================================================
# CONFIGURACIÓN PRINCIPAL
# ============================================================

TEST_H5_PATH = "/data/vimunoz/Symmetria-Intermediate-2-10k-preproc/test.h5"
CHECKPOINT_PATH = "/home/vimunoz/proyectos/PCT_Symmetria/logs/intermediate-2-100k-64-MHA-Optimized-HLoss/version_1/checkpoints/prueba.ckpt"
MODEL_CLASS_PATH = "src.model.LightningSymmetryModel.LightningSymmetryModel"

# -----------------------------
# PARÁMETROS DE EVALUACIÓN
# -----------------------------
CONFIDENCE_THRESHOLD = 0.5
ANGLE_THRESHOLD = 1.0
EPSILON_RATE = 0.01
DBSCAN_EPS = 0.005
DBSCAN_MIN_SAMPLES = 10

# ============================================================
# MODO DE EJECUCIÓN / FIDELIDAD
# ============================================================
# Si quieres máxima fidelidad, parte con:
#   BATCH_SIZE = 1 o 4
#   USE_AMP = False
#   USE_TORCH_COMPILE = False
#   USE_TF32 = False
#   MAX_CANDIDATES_FOR_CLUSTERING = None
#   NORMALIZE_NORMALS_BEFORE_DBSCAN = False
#
# Si quieres acelerar gradualmente:
#   1) sube BATCH_SIZE
#   2) prueba USE_TORCH_COMPILE = True
#   3) prueba USE_AMP = True
#   4) recién después juega con MAX_CANDIDATES_FOR_CLUSTERING
#
# Lo importante: si cambia algo, será por estos parámetros explícitos.

BATCH_SIZE = 4

# Aceleraciones numéricas (pueden alterar algo los resultados)
USE_AMP = False
AMP_DTYPE = "float16"          # "float16" o "bfloat16"
USE_TORCH_COMPILE = False
USE_TF32 = False

# Recorte opcional de candidatos ANTES de DBSCAN
# None = no recortar (más fiel)
# Ej: 4096, 2048, 1024, 512 = más rápido pero menos fiel
MAX_CANDIDATES_FOR_CLUSTERING = None

# Por defecto lo dejo como False para ser fiel al comportamiento original.
# Si tus normales ya están bien normalizadas, poner True podría ayudar a estabilidad,
# pero ya es una decisión metodológica.
NORMALIZE_NORMALS_BEFORE_DBSCAN = False

# ============================================================
# I/O Y PERFILADO
# ============================================================

COPY_H5_TO_LOCAL = True
LOCAL_H5_DIR = os.environ.get("SLURM_TMPDIR", "/tmp")

PROFILE_EVERY_N_BATCHES = 5
PROFILE_POSTPROCESS_BREAKDOWN = True
RESULTS_BASE_DIR = "resultados_evaluacion"

# Para pruebas rápidas
MAX_SHAPES = None   # None = todas. Ej: 200 para benchmark corto


MAX_SHAPES = 300
BATCH_SIZE = 8
USE_AMP = False
USE_TORCH_COMPILE = False
USE_TF32 = False
MAX_CANDIDATES_FOR_CLUSTERING = None
CONFIDENCE_THRESHOLD = 0.20
NORMALIZE_NORMALS_BEFORE_DBSCAN = False
# ============================================================
# UTILIDADES
# ============================================================

def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def parse_amp_dtype(dtype_str: str):
    dtype_str = dtype_str.lower()
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"AMP_DTYPE no soportado: {dtype_str}")

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def safe_category_from_shape_id(shape_id: str) -> str:
    try:
        return shape_id.split("-")[1]
    except Exception:
        return "desconocido"

def maybe_copy_to_local(src_path: str, local_dir: str) -> str:
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"No existe el H5: {src_path}")

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    dst = local_dir / src.name

    if dst.exists():
        try:
            if dst.stat().st_size == src.stat().st_size:
                print(f"📦 Usando copia local existente: {dst}")
                return str(dst)
        except Exception:
            pass

    print(f"📥 Copiando H5 a disco local...\n   Origen: {src}\n   Destino: {dst}")
    t0 = time.perf_counter()
    shutil.copy2(src, dst)
    dt = time.perf_counter() - t0
    print(f"✅ Copia completada en {dt:.2f} s")
    return str(dst)

def get_hparams_from_checkpoint_path(ckpt_path: str) -> dict:
    ckpt = Path(ckpt_path).resolve()
    candidates = []

    for parent in [ckpt.parent, *ckpt.parents[:6]]:
        candidates.append(parent / "hparams.yaml")

    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    hparams = yaml.safe_load(f)
                print(f"📄 hparams.yaml cargado desde: {c}")
                return hparams if isinstance(hparams, dict) else {}
            except Exception as e:
                print(f"⚠️ Error leyendo {c}: {e}")

    print("⚠️ No se encontró hparams.yaml")
    return {}

def load_model_dynamically(class_path: str, ckpt_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    # Carga primero en CPU para evitar comportamientos raros durante el load.
    model = model_class.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model

# ============================================================
# POSTPROCESO
# ============================================================

def extract_final_symmetries(
    pred_normals: torch.Tensor,      # [1, N, M, 3] esperado por tu versión reciente
    pred_confs: torch.Tensor,        # [1, N, M]
    pred_centers: torch.Tensor,      # [1, N, 3]
    conf_threshold: float,
    eps: float,
    min_samples: int,
    max_candidates: int | None = None,
    normalize_normals_before_dbscan: bool = False,
    profile_breakdown: bool = False,
):
    """
    Mantiene la lógica base de tu script:
    - flatten completo
    - threshold por confianza
    - centros expandidos por candidato
    - distancia = 1 - abs(dot_products)
    - DBSCAN(metric='precomputed')
    - mejor candidato por cluster = mayor confianza

    Aceleración segura:
    - el filtrado se hace en torch antes de pasar a CPU

    Aceleración opcional:
    - max_candidates (top-k por confianza antes de DBSCAN)
      Esto sí puede cambiar resultados, pero queda completamente controlado por parámetro.
    """

    breakdown = {
        "filter_and_transfer": 0.0,
        "distance_matrix": 0.0,
        "dbscan": 0.0,
        "select_best": 0.0,
    }

    M = pred_normals.shape[2]

    t0 = time.perf_counter()

    confs_flat = pred_confs.reshape(-1)
    mask = confs_flat > conf_threshold
    num_valid_before = int(mask.sum().item())

    if num_valid_before == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            num_valid_before,
            0,
            breakdown,
        )

    normals_flat = pred_normals.reshape(-1, 3)

    centers_flat = (
        pred_centers.unsqueeze(2)
        .expand(-1, -1, M, -1)
        .reshape(-1, 3)
    )

    valid_idx = torch.where(mask)[0]

    # Recorte opcional controlado por parámetro
    if max_candidates is not None and valid_idx.numel() > max_candidates:
        valid_confs_t = confs_flat[valid_idx]
        _, top_pos = torch.topk(valid_confs_t, k=max_candidates, largest=True, sorted=False)
        valid_idx = valid_idx[top_pos]

    num_valid_after = int(valid_idx.numel())

    valid_normals_t = normals_flat[valid_idx]
    valid_confs_t = confs_flat[valid_idx]
    valid_centers_t = centers_flat[valid_idx]

    if normalize_normals_before_dbscan:
        valid_normals_t = torch.nn.functional.normalize(valid_normals_t, dim=1)

    valid_normals = valid_normals_t.detach().cpu().numpy()
    valid_confs = valid_confs_t.detach().cpu().numpy()
    valid_centers = valid_centers_t.detach().cpu().numpy()

    breakdown["filter_and_transfer"] += time.perf_counter() - t0

    if len(valid_normals) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            num_valid_before,
            num_valid_after,
            breakdown,
        )

    # Matriz de distancias exacta (fiel a tu lógica)
    t0 = time.perf_counter()
    dot_products = np.abs(valid_normals @ valid_normals.T)
    np.clip(dot_products, 0.0, 1.0, out=dot_products)
    distance_matrix = 1.0 - dot_products
    breakdown["distance_matrix"] += time.perf_counter() - t0

    # DBSCAN exacto
    t0 = time.perf_counter()
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
        n_jobs=-1,
    ).fit(distance_matrix)
    labels = clustering.labels_
    breakdown["dbscan"] += time.perf_counter() - t0

    # Elegir mejor por cluster
    t0 = time.perf_counter()
    final_normals, final_confs, final_centers = [], [], []

    for label in set(labels):
        if label == -1:
            continue

        cluster_mask = (labels == label)
        cluster_normals = valid_normals[cluster_mask]
        cluster_confs = valid_confs[cluster_mask]
        cluster_centers = valid_centers[cluster_mask]

        best_idx = np.argmax(cluster_confs)
        final_normals.append(cluster_normals[best_idx])
        final_confs.append(cluster_confs[best_idx])
        final_centers.append(cluster_centers[best_idx])

    breakdown["select_best"] += time.perf_counter() - t0

    return (
        np.asarray(final_normals, dtype=np.float32),
        np.asarray(final_confs, dtype=np.float32),
        np.asarray(final_centers, dtype=np.float32),
        num_valid_before,
        num_valid_after,
        breakdown,
    )

# ============================================================
# PERFILADO
# ============================================================

def print_running_profile(
    timers: dict,
    n_shapes_done: int,
    n_batches_done: int,
    device: torch.device,
    candidate_stats: dict,
):
    if n_shapes_done == 0:
        return

    print("\n" + "=" * 90)
    print("⏱️ PERFIL PARCIAL")
    print(f"Shapes procesadas: {n_shapes_done}")
    print(f"Batches procesados: {n_batches_done}")
    print("-" * 90)

    ordered = [
        "io_h5",
        "tensor_prep_to_device",
        "forward",
        "postprocess_total",
        "postprocess_filter_and_transfer",
        "postprocess_distance_matrix",
        "postprocess_dbscan",
        "postprocess_select_best",
        "pack_predictions",
    ]

    for k in ordered:
        if k in timers:
            print(f"{k:<32}: {timers[k] / n_shapes_done:.4f} s/shape")

    print("-" * 90)
    print(f"avg_valid_candidates_before      : {candidate_stats['sum_before'] / max(candidate_stats['count'], 1):.2f}")
    print(f"avg_valid_candidates_after       : {candidate_stats['sum_after'] / max(candidate_stats['count'], 1):.2f}")

    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print("-" * 90)
        print(f"GPU mem allocada : {mem_alloc:.2f} GB")
        print(f"GPU mem reservada: {mem_reserved:.2f} GB")

    print("=" * 90 + "\n")

# ============================================================
# MAIN
# ============================================================

def main():
    global TEST_H5_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = parse_amp_dtype(AMP_DTYPE)

    # Opciones numéricas controladas
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = USE_TF32
        torch.backends.cudnn.allow_tf32 = USE_TF32

    print(f"🧠 Cargando modelo en {device}...")
    model = load_model_dynamically(MODEL_CLASS_PATH, CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    if device.type == "cuda" and USE_TORCH_COMPILE:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ torch.compile activado")
        except Exception as e:
            print(f"⚠️ No se pudo activar torch.compile: {e}")

    hparams_dict = get_hparams_from_checkpoint_path(CHECKPOINT_PATH)

    if COPY_H5_TO_LOCAL:
        TEST_H5_PATH = maybe_copy_to_local(TEST_H5_PATH, LOCAL_H5_DIR)

    theta_cos = 1.0 - np.cos(np.radians(ANGLE_THRESHOLD))
    pdict = {
        "eps": EPSILON_RATE,
        "theta": theta_cos,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "rot_angle_threshold": 0.01,
    }

    predictions_list = []
    predictions_by_category = defaultdict(list)
    category_metrics_log = {}

    timers = defaultdict(float)
    candidate_stats = {
        "sum_before": 0,
        "sum_after": 0,
        "count": 0,
    }

    total_t0 = time.perf_counter()

    print(f"📂 Procesando: {TEST_H5_PATH}")
    with h5py.File(TEST_H5_PATH, "r") as f:
        shape_ids = list(f.keys())

        if MAX_SHAPES is not None:
            shape_ids = shape_ids[:MAX_SHAPES]

        num_shapes = len(shape_ids)
        num_batches = math.ceil(num_shapes / BATCH_SIZE)

        print(f"🔢 Total de shapes: {num_shapes}")
        print(f"📦 Batch size: {BATCH_SIZE}")
        print(f"🧪 AMP: {USE_AMP} ({AMP_DTYPE})")
        print(f"⚙️ torch.compile: {USE_TORCH_COMPILE}")
        print(f"⚙️ TF32: {USE_TF32}")
        print(f"✂️ Max candidates: {MAX_CANDIDATES_FOR_CLUSTERING}")
        print(f"📐 Normalize normals before DBSCAN: {NORMALIZE_NORMALS_BEFORE_DBSCAN}")

        with torch.inference_mode():
            pbar = tqdm(
                chunked(shape_ids, BATCH_SIZE),
                total=num_batches,
                desc="Inferencia + Clustering",
            )

            for batch_idx, batch_ids in enumerate(pbar, start=1):
                # ------------------------------------------------
                # 1) LECTURA H5
                # ------------------------------------------------
                t0 = time.perf_counter()

                batch_points_np = []
                batch_gt_np = []
                batch_categories = []

                for shape_id in batch_ids:
                    group = f[shape_id]
                    points_np = group["points"][:].astype(np.float32, copy=False)

                    if "planar_symmetries" in group:
                        gt_planes = group["planar_symmetries"][:].astype(np.float32, copy=False)
                    else:
                        gt_planes = np.empty((0, 6), dtype=np.float32)

                    batch_points_np.append(points_np)
                    batch_gt_np.append(gt_planes)
                    batch_categories.append(safe_category_from_shape_id(shape_id))

                timers["io_h5"] += time.perf_counter() - t0

                # ------------------------------------------------
                # 2) PREPARAR TENSORES
                # ------------------------------------------------
                t0 = time.perf_counter()

                points_np_batch = np.stack(batch_points_np, axis=0)          # [B, N, 3]
                points_batch_cpu = torch.from_numpy(points_np_batch)          # [B, N, 3]
                points_batch_model = points_batch_cpu.permute(0, 2, 1).contiguous()  # [B, 3, N]

                if device.type == "cuda":
                    points_batch_model = points_batch_model.pin_memory().to(device, non_blocking=True)
                else:
                    points_batch_model = points_batch_model.to(device)

                timers["tensor_prep_to_device"] += time.perf_counter() - t0

                # ------------------------------------------------
                # 3) FORWARD
                # ------------------------------------------------
                t0 = time.perf_counter()

                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if (device.type == "cuda" and USE_AMP)
                    else nullcontext()
                )

                with amp_ctx:
                    pred_n, pred_c, pred_cent = model(points_batch_model)

                cuda_sync(device)
                timers["forward"] += time.perf_counter() - t0

                # ------------------------------------------------
                # 4) POSTPROCESO POR SHAPE
                # ------------------------------------------------
                batch_size_real = len(batch_ids)

                for b in range(batch_size_real):
                    t0_post = time.perf_counter()

                    final_n, final_c, final_cent, n_before, n_after, breakdown = extract_final_symmetries(
                        pred_n[b:b+1],
                        pred_c[b:b+1],
                        pred_cent[b:b+1],
                        conf_threshold=CONFIDENCE_THRESHOLD,
                        eps=DBSCAN_EPS,
                        min_samples=DBSCAN_MIN_SAMPLES,
                        max_candidates=MAX_CANDIDATES_FOR_CLUSTERING,
                        normalize_normals_before_dbscan=NORMALIZE_NORMALS_BEFORE_DBSCAN,
                        profile_breakdown=PROFILE_POSTPROCESS_BREAKDOWN,
                    )

                    timers["postprocess_total"] += time.perf_counter() - t0_post
                    timers["postprocess_filter_and_transfer"] += breakdown["filter_and_transfer"]
                    timers["postprocess_distance_matrix"] += breakdown["distance_matrix"]
                    timers["postprocess_dbscan"] += breakdown["dbscan"]
                    timers["postprocess_select_best"] += breakdown["select_best"]

                    candidate_stats["sum_before"] += n_before
                    candidate_stats["sum_after"] += n_after
                    candidate_stats["count"] += 1

                    # ----------------------------
                    # empaquetado para métricas
                    # ----------------------------
                    t0_pack = time.perf_counter()

                    if len(final_n) > 0:
                        y_pred_tensor = torch.cat([
                            torch.from_numpy(final_n).to(torch.float32),
                            torch.from_numpy(final_cent).to(torch.float32),
                            torch.from_numpy(final_c).to(torch.float32).unsqueeze(1),
                        ], dim=1)
                    else:
                        y_pred_tensor = torch.empty((0, 7), dtype=torch.float32)

                    gt_planes_np = batch_gt_np[b]
                    if len(gt_planes_np) > 0:
                        y_true_tensor = torch.from_numpy(gt_planes_np).to(torch.float32)
                    else:
                        y_true_tensor = torch.empty((0, 6), dtype=torch.float32)

                    prediction_item = [
                        points_batch_cpu[b:b+1].contiguous(),   # [1, N, 3]
                        y_pred_tensor.unsqueeze(0).cpu(),       # [1, P, 7]
                        [y_true_tensor.cpu()],
                    ]

                    predictions_list.append(prediction_item)
                    predictions_by_category[batch_categories[b]].append(prediction_item)

                    timers["pack_predictions"] += time.perf_counter() - t0_pack

                done = len(predictions_list)
                elapsed = time.perf_counter() - total_t0
                avg_per_shape = elapsed / max(done, 1)
                eta = avg_per_shape * (num_shapes - done)

                pbar.set_postfix({
                    "done": f"{done}/{num_shapes}",
                    "avg_s/shape": f"{avg_per_shape:.2f}",
                    "eta_h": f"{eta / 3600:.2f}",
                    "cand_before": f"{candidate_stats['sum_before'] / max(candidate_stats['count'], 1):.0f}",
                    "cand_after": f"{candidate_stats['sum_after'] / max(candidate_stats['count'], 1):.0f}",
                })

                if batch_idx % PROFILE_EVERY_N_BATCHES == 0:
                    print_running_profile(
                        timers,
                        n_shapes_done=done,
                        n_batches_done=batch_idx,
                        device=device,
                        candidate_stats=candidate_stats,
                    )

    timers["total_runtime"] = time.perf_counter() - total_t0

    # ============================================================
    # MÉTRICAS GLOBALES
    # ============================================================
    print("\n📊 Calculando métricas globales...")
    t0 = time.perf_counter()

    total_map, total_phc, _ = calculate_metrics_from_predictions(
        predictions_list,
        get_match_sequence_plane_symmetry,
        pdict,
    )

    timers["global_metrics"] = time.perf_counter() - t0

    map_val = total_map.item()
    phc_val = total_phc.item()

    print("=" * 70)
    print(f"🥇 Official Paper mAP (Global): {map_val:.4f}")
    print(f"🥇 Official Paper PHC (Global): {phc_val:.4f}")
    print("=" * 70)

    # ============================================================
    # MÉTRICAS POR CATEGORÍA
    # ============================================================
    print("\n" + "=" * 70)
    print(f"{'RENDIMIENTO POR CATEGORÍA':^70}")
    print("=" * 70)
    print(f"{'Categoría':<25} | {'mAP':<8} | {'PHC':<8} | {'Nº Figuras'}")
    print("-" * 70)

    t0 = time.perf_counter()

    for cat in sorted(predictions_by_category.keys()):
        cat_preds = predictions_by_category[cat]

        cat_map, cat_phc, _ = calculate_metrics_from_predictions(
            cat_preds,
            get_match_sequence_plane_symmetry,
            pdict,
        )

        c_map_val = cat_map.item()
        c_phc_val = cat_phc.item()

        category_metrics_log[cat] = {
            "mAP": round(c_map_val, 4),
            "PHC": round(c_phc_val, 4),
            "count": len(cat_preds),
        }

        print(f"{cat:<25} | {c_map_val:.4f}   | {c_phc_val:.4f}   | {len(cat_preds)}")

    timers["category_metrics"] = time.perf_counter() - t0

    print("=" * 70 + "\n")

    # ============================================================
    # RESUMEN
    # ============================================================
    print("=" * 90)
    print("⏱️ RESUMEN FINAL")
    print("=" * 90)

    ordered = [
        "io_h5",
        "tensor_prep_to_device",
        "forward",
        "postprocess_total",
        "postprocess_filter_and_transfer",
        "postprocess_distance_matrix",
        "postprocess_dbscan",
        "postprocess_select_best",
        "pack_predictions",
        "global_metrics",
        "category_metrics",
        "total_runtime",
    ]

    for k in ordered:
        if k in timers:
            print(f"{k:<32}: {timers[k]:.4f} s")

    if len(predictions_list) > 0:
        print("-" * 90)
        print(f"{'promedio_total_por_shape':<32}: {timers['total_runtime'] / len(predictions_list):.6f} s")
        print(f"{'avg_valid_candidates_before':<32}: {candidate_stats['sum_before'] / max(candidate_stats['count'], 1):.2f}")
        print(f"{'avg_valid_candidates_after':<32}: {candidate_stats['sum_after'] / max(candidate_stats['count'], 1):.2f}")

    print("=" * 90)

    # ============================================================
    # GUARDAR RESULTADOS
    # ============================================================
    experiment_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": CHECKPOINT_PATH,
        "dataset_test": TEST_H5_PATH,
        "hparams": hparams_dict,
        "runtime": {
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "use_amp": USE_AMP,
            "amp_dtype": AMP_DTYPE,
            "use_torch_compile": USE_TORCH_COMPILE,
            "use_tf32": USE_TF32,
            "copy_h5_to_local": COPY_H5_TO_LOCAL,
            "max_candidates_for_clustering": MAX_CANDIDATES_FOR_CLUSTERING,
            "normalize_normals_before_dbscan": NORMALIZE_NORMALS_BEFORE_DBSCAN,
            "timers_seconds": {k: round(v, 6) for k, v in timers.items()},
        },
        "candidate_stats": {
            "avg_valid_candidates_before": round(candidate_stats["sum_before"] / max(candidate_stats["count"], 1), 2),
            "avg_valid_candidates_after": round(candidate_stats["sum_after"] / max(candidate_stats["count"], 1), 2),
        },
        "parameters": {
            "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
            "ANGLE_THRESHOLD": ANGLE_THRESHOLD,
            "EPSILON_RATE": EPSILON_RATE,
            "DBSCAN_EPS": DBSCAN_EPS,
            "DBSCAN_MIN_SAMPLES": DBSCAN_MIN_SAMPLES,
        },
        "official_metrics": {
            "mAP": round(map_val, 4),
            "PHC": round(phc_val, 4),
        },
        "metrics_by_category": category_metrics_log,
    }

    dataset_name = Path(TEST_H5_PATH).parent.name
    base_dir = Path(RESULTS_BASE_DIR) / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    map_short = f"{map_val:.2f}"
    phc_short = f"{phc_val:.2f}"
    safe_time_str = datetime.now().strftime("%d%m%y_%H%M")

    file_name = base_dir / f"eval_mAP{map_short}_PHC{phc_short}_{safe_time_str}.json"

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=4, ensure_ascii=False)

    print(f"💾 Resultados guardados en: {file_name.resolve()}")

if __name__ == "__main__":
    main()