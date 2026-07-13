import os
import re
import torch
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from pathlib import Path
import importlib
import json
from datetime import datetime
import yaml  # <-- NUEVA IMPORTACIÓN

# Asegúrate de que estas rutas de importación coincidan con la estructura de tu proyecto
from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry

# ==========================================
# CONFIGURACIÓN DE PARÁMETROS
# ==========================================

TEST_H5_PATH = "/data/vimunoz/Symmetria-Intermediate-2-100k-preproc/test.h5"
CHECKPOINT_PATH = "/home/vimunoz/PCT_Symmetria/logs/ablation-E20-mha-standard-hard100k-rsd03/version_1/checkpoints/best-epoch=198-val_loss=0.00542.ckpt"
# Opciones: "highest_epoch" o "lowest_val_loss"
BEST_CHECKPOINT_SELECTION = "highest_epoch"

#CHECKPOINT_PATH = "/home/vimunoz/proyectos/PCT_Symmetria/logs/hard-100k-64-MHA-StandardAttn-HLoss/version_0/checkpoints/last.ckpt"
MODEL_CLASS_PATH = "src.model.LightningSymmetryModel.LightningSymmetryModel"

CONFIDENCE_THRESHOLD = 0.9
ANGLE_THRESHOLD = 1.0       # Grados permitidos de ferror
EPSILON_RATE = 0.01         # Porcentaje de la diagonal para el error de distancia
DBSCAN_EPS = 0.015         # Umbral de distancia Coseno
DBSCAN_MIN_SAMPLES = 10

def load_model_dynamically(class_path, ckpt_path):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    
    #target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_device = torch.device('cpu')
    #torch.cuda.set_device(1) # Forzado a la GPU 1 según tu configuración
    return model_class.load_from_checkpoint(ckpt_path, map_location=target_device)

def find_best_checkpoint_for_version(ckpt_path, selection="highest_epoch"):
    """
    Busca checkpoints tipo:
        best-epoch=158-val_loss=0.00529.ckpt
    dentro de la misma carpeta checkpoints/ del CHECKPOINT_PATH usado.

    selection:
        - "highest_epoch": elige el best checkpoint con mayor época.
        - "lowest_val_loss": elige el best checkpoint con menor val_loss.
    """
    checkpoints_dir = Path(ckpt_path).expanduser().resolve().parent
    regex = re.compile(r"^best-epoch=(\d+)-val_loss=([0-9]*\.?[0-9]+)\.ckpt$")

    candidates = []
    for path in checkpoints_dir.glob("best-epoch=*-val_loss=*.ckpt"):
        match = regex.match(path.name)
        if not match:
            continue

        epoch = int(match.group(1))
        val_loss = float(match.group(2))
        candidates.append({
            "path": path,
            "epoch": epoch,
            "val_loss": val_loss,
        })

    if not candidates:
        print(f"⚠️ No se encontraron checkpoints best-* en: {checkpoints_dir}")
        return None

    if selection == "lowest_val_loss":
        # Si hay empate en loss, preferimos la época mayor.
        best = min(candidates, key=lambda x: (x["val_loss"], -x["epoch"]))
    elif selection == "highest_epoch":
        # Si hay empate en época, preferimos el menor loss.
        best = max(candidates, key=lambda x: (x["epoch"], -x["val_loss"]))
    else:
        raise ValueError(
            f"BEST_CHECKPOINT_SELECTION inválido: {selection}. "
            "Usa 'highest_epoch' o 'lowest_val_loss'."
        )

    return str(best["path"])

def compute_detailed_metrics(pred_normals, gt_planes, angle_threshold=1.0):
    """
    Calcula TP, FP, FN, Precision, Recall y MAE (Mean Angular Error).
    pred_normals: (N, 3) numpy array
    gt_planes: (K, 6) numpy array donde K[:, :3] son las normales reales
    """
    if len(gt_planes) == 0:
        # Si no hay GT (ej. Revolution) y predecimos algo, son puros Falsos Positivos
        FP = len(pred_normals)
        return {"TP": 0, "FP": FP, "FN": 0, "MAE": 0.0}
    
    if len(pred_normals) == 0:
        # Si hay GT pero no predecimos nada, son puros Falsos Negativos
        return {"TP": 0, "FP": 0, "FN": len(gt_planes), "MAE": 0.0}

    gt_normals = gt_planes[:, :3]
    
    # Producto punto y conversión a grados
    dot_products = np.abs(np.dot(pred_normals, gt_normals.T))
    dot_products = np.clip(dot_products, 0.0, 1.0)
    angle_errors = np.degrees(np.arccos(dot_products)) # Matriz (N, K)
    
    TP, FP, FN = 0, 0, 0
    angular_errors_tp = []
    
    # Bipartite matching simple para evaluación de métricas de diagnóstico
    matched_gt = set()
    for i in range(len(pred_normals)):
        best_gt_idx = np.argmin(angle_errors[i])
        best_error = angle_errors[i, best_gt_idx]
        
        if best_error <= angle_threshold and best_gt_idx not in matched_gt:
            TP += 1
            matched_gt.add(best_gt_idx)
            angular_errors_tp.append(best_error)
        else:
            FP += 1 # Predicción ruidosa o el GT ya fue tomado por una predicción mejor
            
    FN = len(gt_planes) - len(matched_gt) # Planos reales que nadie encontró
    
    mae = float(np.mean(angular_errors_tp)) if len(angular_errors_tp) > 0 else 0.0
    
    return {"TP": TP, "FP": FP, "FN": FN, "MAE": mae}

def extract_final_symmetries_cosine(pred_normals, pred_confs, pred_centers, conf_threshold, eps, min_samples):
    M = pred_normals.shape[2] 
    normals_flat = pred_normals.reshape(-1, 3).cpu().numpy()
    confs_flat = pred_confs.reshape(-1).cpu().numpy()
    
    centers_expanded = pred_centers.unsqueeze(2).expand(-1, -1, M, -1)
    centers_flat = centers_expanded.reshape(-1, 3).cpu().numpy()
    
    mask = confs_flat > conf_threshold
    valid_normals = normals_flat[mask]
    valid_confs = confs_flat[mask]
    valid_centers = centers_flat[mask]
    
    if len(valid_normals) == 0:
        return np.array([]), np.array([]), np.array([])
        
    dot_products = np.abs(np.dot(valid_normals, valid_normals.T))
    dot_products = np.clip(dot_products, 0.0, 1.0)
    distance_matrix = 1.0 - dot_products
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
    labels = clustering.labels_
    
    final_normals, final_confs, final_centers = [], [], []
    for label in set(labels):
        if label == -1: continue 
            
        cluster_mask = (labels == label)
        cluster_normals = valid_normals[cluster_mask]
        cluster_confs = valid_confs[cluster_mask]
        cluster_centers = valid_centers[cluster_mask]
        
        best_idx = np.argmax(cluster_confs)
        final_normals.append(cluster_normals[best_idx])
        final_confs.append(cluster_confs[best_idx])
        final_centers.append(cluster_centers[best_idx])
        
    return np.array(final_normals), np.array(final_confs), np.array(final_centers)

def main():
    print(f"🧠 Cargando modelo dinámicamente...")
    model = load_model_dynamically(MODEL_CLASS_PATH, CHECKPOINT_PATH)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"⚡ Evaluando usando VRAM: {device}\n")
    
    # --- NUEVO: Cargar hparams.yaml ---
    hparams_data = {}
    ckpt_path_obj = Path(CHECKPOINT_PATH)

    # Buscar el mejor checkpoint dentro de la misma carpeta checkpoints/ de esta version_x
    best_model_checkpoint = find_best_checkpoint_for_version(
        CHECKPOINT_PATH,
        selection=BEST_CHECKPOINT_SELECTION
    )
    if best_model_checkpoint is not None:
        print(f"🏆 Best checkpoint detectado ({BEST_CHECKPOINT_SELECTION}): {best_model_checkpoint}\n")

    # Retrocedemos dos niveles: version_x/checkpoints/last.ckpt -> version_x/hparams.yaml
    hparams_path = ckpt_path_obj.parent.parent / "hparams.yaml"
    
    if hparams_path.exists():
        try:
            with open(hparams_path, 'r', encoding='utf-8') as f:
                hparams_data = yaml.safe_load(f)
            print(f"📄 hparams.yaml cargado exitosamente desde: {hparams_path}\n")
        except Exception as e:
            print(f"⚠️ Error al leer hparams.yaml: {e}\n")
    else:
        print(f"⚠️ Advertencia: No se encontró hparams.yaml en {hparams_path}\n")
    # -----------------------------------
    
    theta_cos = 1.0 - np.cos(np.radians(ANGLE_THRESHOLD)) 
    pdict = {
        "eps": EPSILON_RATE,
        "theta": theta_cos,  
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "rot_angle_threshold": 0.01
    }
    
    predictions_list = []
    predictions_by_category = {} 

    print(f"📂 Abriendo base de datos HDF5: {TEST_H5_PATH}")
    with h5py.File(TEST_H5_PATH, 'r') as f:
        shape_ids = list(f.keys())
        
        with torch.no_grad():
            for shape_id in tqdm(shape_ids, desc="Generando Predicciones"):
                group = f[shape_id]
                points_np = group['points'][:]
                gt_planes = group['planar_symmetries'][:] if 'planar_symmetries' in group else []
                
                t_points = torch.tensor(points_np, dtype=torch.float32)
                points_tensor = t_points.transpose(0, 1).unsqueeze(0).to(device)
                
                pred_n, pred_c, pred_cent = model(points_tensor)

                final_n, final_c, final_cent = extract_final_symmetries_cosine(
                    pred_n, pred_c, pred_cent, 
                    conf_threshold=CONFIDENCE_THRESHOLD,
                    eps=DBSCAN_EPS,
                    min_samples=DBSCAN_MIN_SAMPLES
                )
                
                gt_planes_np = np.array(gt_planes) if len(gt_planes) > 0 else np.empty((0, 6))
                detailed_stats = compute_detailed_metrics(final_n, gt_planes_np, angle_threshold=ANGLE_THRESHOLD)
                
                if len(final_n) > 0:
                    y_pred_tensor = torch.cat([
                        torch.tensor(final_n, dtype=torch.float32),
                        torch.tensor(final_cent, dtype=torch.float32),
                        torch.tensor(final_c, dtype=torch.float32).unsqueeze(1)
                    ], dim=1)
                else:
                    y_pred_tensor = torch.empty((0, 7), dtype=torch.float32)
                
                if len(gt_planes) > 0:
                    y_true_tensor = torch.tensor(gt_planes, dtype=torch.float32)
                else:
                    y_true_tensor = torch.empty((0, 6), dtype=torch.float32)
                    
                prediction_item = [
                    points_tensor.transpose(1, 2).cpu(), 
                    y_pred_tensor.unsqueeze(0).cpu(),    
                    [y_true_tensor.cpu()]                
                ]
                
                predictions_list.append(prediction_item)
                
                try:
                    categoria = shape_id.split('-')[1]
                except IndexError:
                    categoria = "desconocido" 
                
                if categoria not in predictions_by_category:
                    predictions_by_category[categoria] = {'items': [], 'stats': []}
                    
                predictions_by_category[categoria]['items'].append(prediction_item)
                predictions_by_category[categoria]['stats'].append(detailed_stats)

    # --- MÉTRICAS GLOBALES ---
    print("\n📊 Calculando métricas oficiales del paper...")
    total_map, total_phc, _ = calculate_metrics_from_predictions(
        predictions_list, 
        get_match_sequence_plane_symmetry, 
        pdict
    )
    
    map_val = total_map.item()
    phc_val = total_phc.item()

    print("="*95)
    print(f"🥇 Official Paper mAP (Global): {map_val:.4f}")
    print(f"🥇 Official Paper PHC (Global): {phc_val:.4f}")
    print("="*95)

    # --- MÉTRICAS DETALLADAS POR CATEGORÍA ---
    print("\n" + "="*95)
    print(f"{'RENDIMIENTO DETALLADO POR CATEGORÍA':^95}")
    print("="*95)
    print(f"{'Categoría':<20} | {'mAP':<6} | {'PHC':<6} | {'Prec.':<6} | {'Recall':<6} | {'MAE (Grados)':<12} | {'FP / Fig'}")
    print("-" * 95)

    category_metrics_log = {}

    for cat in sorted(predictions_by_category.keys()):
        cat_data = predictions_by_category[cat]
        cat_preds = cat_data['items']
        cat_stats = cat_data['stats']
        
        # Oficiales
        cat_map, cat_phc, _ = calculate_metrics_from_predictions(cat_preds, get_match_sequence_plane_symmetry, pdict)
        c_map_val = cat_map.item()
        c_phc_val = cat_phc.item()
        
        # Sumatorias de TP, FP, FN
        sum_tp = sum(s["TP"] for s in cat_stats)
        sum_fp = sum(s["FP"] for s in cat_stats)
        sum_fn = sum(s["FN"] for s in cat_stats)
        
        # Promedio MAE solo para aciertos
        maes = [s["MAE"] for s in cat_stats if s["MAE"] > 0]
        avg_mae = float(np.mean(maes)) if len(maes) > 0 else 0.0
        
        precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
        recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
        avg_fp_per_shape = sum_fp / len(cat_preds)
        
        # Guardado enriquecido para el JSON
        category_metrics_log[cat] = {
            "mAP": round(c_map_val, 4),
            "PHC": round(c_phc_val, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "MAE_Grados": round(avg_mae, 4),
            "FP_per_shape": round(avg_fp_per_shape, 2),
            "count": len(cat_preds)
        }
        
        print(f"{cat:<20} | {c_map_val:.4f} | {c_phc_val:.4f} | {precision:.4f} | {recall:.4f} | {avg_mae:.4f}°       | {avg_fp_per_shape:.2f}")

    print("="*95 + "\n")

    # ==========================================
    # GUARDAR RESULTADOS EN JSON (ACTUALIZADO)
    # ==========================================
    experiment_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": CHECKPOINT_PATH,
        "best_model_checkpoint": best_model_checkpoint,
        "hparams": hparams_data, # <-- NUEVO: Aquí se inyectan los hiperparámetros
        "dataset_test": TEST_H5_PATH,
        "parameters": {
            "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
            "ANGLE_THRESHOLD": ANGLE_THRESHOLD,
            "EPSILON_RATE": EPSILON_RATE,
            "DBSCAN_EPS": DBSCAN_EPS,
            "DBSCAN_MIN_SAMPLES": DBSCAN_MIN_SAMPLES
        },
        "official_metrics": {
            "mAP": round(map_val, 4),
            "PHC": round(phc_val, 4)
        },
        "metrics_by_category": category_metrics_log 
    }

    # 1. Extraer nombre del dataset para la carpeta
    dataset_name = TEST_H5_PATH.split('/')[-2]
    base_dir = Path("resultados_evaluacion") / dataset_name
    os.makedirs(base_dir, exist_ok=True)

    # 2. Formatear métricas y fecha para el nombre del archivo
    map_short = f"{map_val:.2f}"
    phc_short = f"{phc_val:.2f}"
    safe_time_str = datetime.now().strftime('%d%m%y_%H%M')

    # 3. Construir el nombre del archivo
    file_name = base_dir / f"eval_mAP{map_short}_PHC{phc_short}_{safe_time_str}.json"

    # 4. Guardar archivo
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=4)

    print(f"💾 Resultados guardados exitosamente en: {file_name.resolve()}")

if __name__ == "__main__":
    main()