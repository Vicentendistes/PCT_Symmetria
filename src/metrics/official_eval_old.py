import os
import torch
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from pathlib import Path
import importlib
from collections import defaultdict

import json
from datetime import datetime

# Asegúrate de que estas rutas de importación coincidan con la estructura de tu proyecto
from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry

# ==========================================
# CONFIGURACIÓN DE PARÁMETROS
# ==========================================

TEST_H5_PATH = "/data/vimunoz/Symmetria-Hard-100k-preproc/test.h5"
CHECKPOINT_PATH = "/home/vimunoz/proyectos/PCT_Symmetria/logs/hard-100k-64-MHA-Optimized-HLoss/version_9/checkpoints/last.ckpt"
MODEL_CLASS_PATH = "src.model.LightningSymmetryModel.LightningSymmetryModel"

CONFIDENCE_THRESHOLD = 0.9
ANGLE_THRESHOLD = 1.0       # Grados permitidos de error
EPSILON_RATE = 0.01         # Porcentaje de la diagonal para el error de distancia
DBSCAN_EPS = 0.005    # Umbral de distancia Coseno (aprox 5 grados)
DBSCAN_MIN_SAMPLES = 10

def load_model_dynamically(class_path, ckpt_path):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    return model_class.load_from_checkpoint(ckpt_path, map_location=target_device)

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
    
    theta_cos = 1.0 - np.cos(np.radians(ANGLE_THRESHOLD)) 
    pdict = {
        "eps": EPSILON_RATE,
        "theta": theta_cos,  
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "rot_angle_threshold": 0.01
    }
    
    predictions_list = []
    predictions_by_category = defaultdict(list) 

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
                
                predictions_by_category[categoria].append(prediction_item)

    # --- MÉTRICAS GLOBALES ---
    print("\n📊 Calculando métricas oficiales del paper...")
    total_map, total_phc, _ = calculate_metrics_from_predictions(
        predictions_list, 
        get_match_sequence_plane_symmetry, 
        pdict
    )
    
    map_val = total_map.item()
    phc_val = total_phc.item()

    print("="*60)
    print(f"🥇 Official Paper mAP (Global): {map_val:.4f}")
    print(f"🥇 Official Paper PHC (Global): {phc_val:.4f}")
    print("="*60)

    # --- MÉTRICAS POR CATEGORÍA ---
    print("\n" + "="*60)
    print(f"{'RENDIMIENTO POR CATEGORÍA':^60}")
    print("="*60)
    print(f"{'Categoría':<25} | {'mAP':<8} | {'PHC':<8} | {'Nº Figuras'}")
    print("-" * 60)

    category_metrics_log = {}

    for cat in sorted(predictions_by_category.keys()):
        cat_preds = predictions_by_category[cat]
        
        cat_map, cat_phc, _ = calculate_metrics_from_predictions(
            cat_preds, 
            get_match_sequence_plane_symmetry, 
            pdict
        )
        
        c_map_val = cat_map.item()
        c_phc_val = cat_phc.item()
        
        category_metrics_log[cat] = {
            "mAP": round(c_map_val, 4),
            "PHC": round(c_phc_val, 4),
            "count": len(cat_preds)
        }
        
        print(f"{cat:<25} | {c_map_val:.4f}   | {c_phc_val:.4f}   | {len(cat_preds)}")

    print("="*60 + "\n")

    # ==========================================
    # GUARDAR RESULTADOS EN JSON (ACTUALIZADO)
    # ==========================================
    experiment_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": CHECKPOINT_PATH,
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
    # Usamos un formato de hora seguro para nombres de archivo: DDMMYY_HHMM
    safe_time_str = datetime.now().strftime('%d%m%y_%H%M')

    # 3. Construir el nombre del archivo
    file_name = base_dir / f"eval_mAP{map_short}_PHC{phc_short}_{safe_time_str}.json"

    # 4. Guardar archivo directamente (sin leer ni hacer append)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=4)

    print(f"💾 Resultados guardados exitosamente en: {file_name.resolve()}")

if __name__ == "__main__":
    main()