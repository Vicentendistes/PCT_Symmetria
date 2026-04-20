import os
import torch
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from pathlib import Path
import importlib
import itertools
import json
from datetime import datetime

# Importaciones de tu proyecto
from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry

# ==========================================
# CONFIGURACIÓN DEL GRID SEARCH
# ==========================================
# Usaremos el dataset de 10k como pediste para mayor velocidad
TEST_H5_PATH = "/data/vimunoz/Symmetria-Hard-10k-preproc/test.h5"
CHECKPOINT_PATH = "/home/vimunoz/proyectos/PCT_Symmetria/logs/hard-100k-64-MHA-Optimized-HLoss/version_0/checkpoints/last.ckpt"
MODEL_CLASS_PATH = "src.model.LightningSymmetryModel.LightningSymmetryModel"

# INTOCABLES (Reglas de evaluación oficial)
ANGLE_THRESHOLD = 1.0       
EPSILON_RATE = 0.01         

# LA GRILLA DE EXPERIMENTOS (Parámetros optimizables)
CONFIDENCE_THRESHOLDS = [0.3, 0.5, 0.7]
DBSCAN_EPSILONS = [0.001, 0.005, 0.01]
DBSCAN_MIN_SAMPLES_LIST = [10, 50, 100]

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def load_model_dynamically(class_path, ckpt_path):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Manejo de error si hay muy pocos puntos para DBSCAN
    if len(distance_matrix) < min_samples:
        return np.array([]), np.array([]), np.array([])

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

# ==========================================
# BUCLE PRINCIPAL
# ==========================================
def main():
    print(f"🧠 Cargando modelo dinámicamente...")
    model = load_model_dynamically(MODEL_CLASS_PATH, CHECKPOINT_PATH)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generar todas las combinaciones de la grilla (3 x 3 x 3 = 27)
    grid = list(itertools.product(CONFIDENCE_THRESHOLDS, DBSCAN_EPSILONS, DBSCAN_MIN_SAMPLES_LIST))
    print(f"⚡ Iniciando Grid Search con {len(grid)} combinaciones en VRAM: {device}\n")
    
    # Diccionario para almacenar las listas de predicciones de cada experimento
    # grid_predictions[idx_experimento] = [prediction_item_1, prediction_item_2, ...]
    grid_predictions = {i: [] for i in range(len(grid))}

    print(f"📂 Abriendo base de datos HDF5: {TEST_H5_PATH}")
    with h5py.File(TEST_H5_PATH, 'r') as f:
        shape_ids = list(f.keys())
        
        with torch.no_grad():
            for shape_id in tqdm(shape_ids, desc="Procesando Inferencia & Clustering"):
                group = f[shape_id]
                points_np = group['points'][:]
                gt_planes = group['planar_symmetries'][:] if 'planar_symmetries' in group else []
                
                t_points = torch.tensor(points_np, dtype=torch.float32)
                points_tensor = t_points.transpose(0, 1).unsqueeze(0).to(device)
                
                # INFERENCIA: Solo la corremos 1 vez por objeto (¡Ahorra horas!)
                pred_n, pred_c, pred_cent = model(points_tensor)

                if len(gt_planes) > 0:
                    y_true_tensor = torch.tensor(gt_planes, dtype=torch.float32).cpu()
                else:
                    y_true_tensor = torch.empty((0, 6), dtype=torch.float32).cpu()

                # CLUSTERING: Aplicamos las 27 configuraciones a la misma inferencia
                for i, (conf, eps, min_s) in enumerate(grid):
                    final_n, final_c, final_cent = extract_final_symmetries_cosine(
                        pred_n, pred_c, pred_cent, 
                        conf_threshold=conf, eps=eps, min_samples=min_s
                    )
                    
                    if len(final_n) > 0:
                        y_pred_tensor = torch.cat([
                            torch.tensor(final_n, dtype=torch.float32),
                            torch.tensor(final_cent, dtype=torch.float32),
                            torch.tensor(final_c, dtype=torch.float32).unsqueeze(1)
                        ], dim=1).cpu()
                    else:
                        y_pred_tensor = torch.empty((0, 7), dtype=torch.float32).cpu()
                    
                    prediction_item = [
                        points_tensor.transpose(1, 2).cpu(), 
                        y_pred_tensor.unsqueeze(0),    
                        [y_true_tensor]                
                    ]
                    
                    grid_predictions[i].append(prediction_item)

    # --- EVALUACIÓN DE LAS 27 COMBINACIONES ---
    print("\n📊 Calculando métricas globales para todas las combinaciones...")
    theta_cos = 1.0 - np.cos(np.radians(ANGLE_THRESHOLD)) 
    pdict = {
        "eps": EPSILON_RATE,
        "theta": theta_cos,  
        "confidence_threshold": 0.0, # Se pone en 0 porque ya filtramos en la extracción
        "rot_angle_threshold": 0.01
    }

    results_summary = []

    for i, (conf, eps, min_s) in enumerate(tqdm(grid, desc="Evaluando métricas")):
        preds = grid_predictions[i]
        
        total_map, total_phc, _ = calculate_metrics_from_predictions(
            preds, get_match_sequence_plane_symmetry, pdict
        )
        
        results_summary.append({
            "conf_threshold": conf,
            "dbscan_eps": eps,
            "min_samples": min_s,
            "mAP": round(total_map.item(), 4),
            "PHC": round(total_phc.item(), 4)
        })

    # Ordenar resultados de mejor a peor mAP
    results_summary.sort(key=lambda x: x["mAP"], reverse=True)

    # Mostrar el Top 5 en consola
    print("\n" + "="*70)
    print(f"{'🏆 TOP 5 MEJORES COMBINACIONES (Dataset Hard 10k)':^70}")
    print("="*70)
    print(f"{'Conf':<6} | {'EPS':<7} | {'MinPts':<7} || {'mAP':<8} | {'PHC':<8}")
    print("-" * 70)
    for res in results_summary[:5]:
        print(f"{res['conf_threshold']:<6} | {res['dbscan_eps']:<7} | {res['min_samples']:<7} || {res['mAP']:<8.4f} | {res['PHC']:<8.4f}")
    print("="*70 + "\n")

    # Guardar reporte completo en JSON
    base_dir = Path("resultados_evaluacion") / "GridSearch"
    os.makedirs(base_dir, exist_ok=True)
    safe_time_str = datetime.now().strftime('%d%m%y_%H%M')
    file_name = base_dir / f"grid_search_hard10k_{safe_time_str}.json"

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)
        
    print(f"💾 Reporte completo de las 27 combinaciones guardado en: {file_name.resolve()}")

if __name__ == "__main__":
    main()