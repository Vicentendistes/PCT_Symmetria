import torch
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from pathlib import Path
import importlib

import json
from datetime import datetime
from pathlib import Path

# Asegúrate de que estas rutas de importación coincidan con la estructura de tu proyecto
from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry


# ==========================================
# CONFIGURACIÓN DE PARÁMETROS
# ==========================================

#TEST_H5_PATH = "/data/vimunoz/Symmetria-Easy-10k-preproc/test.h5"
#CHECKPOINT_PATH = "/home/vimunoz/proyectos/PCT_Symmetria/logs/easy-10k-32/version_1/checkpoints/last.ckpt"
TEST_H5_PATH = "/data/vimunoz/Symmetria-Intermediate-2-100k-preproc/test.h5"
CHECKPOINT_PATH = "/home/vimunoz/proyectos/PCT_Symmetria/logs/intermediate-2-100k-64-MHA-Optimized/version_2/checkpoints/last.ckpt"
MODEL_CLASS_PATH = "src.model.LightningSymmetryModel.LightningSymmetryModel"

CONFIDENCE_THRESHOLD = 0.1
ANGLE_THRESHOLD = 1.0       # Grados permitidos de error
EPSILON_RATE = 0.01         # Porcentaje de la diagonal para el error de distancia
DBSCAN_EPS = 0.005          # Umbral de distancia Coseno (aprox 5 grados)
DBSCAN_MIN_SAMPLES = 10

# Importa tus funciones de carga (ajusta el import según donde las tengas definidas)
def load_model_dynamically(class_path, ckpt_path):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    
    # Detectamos el hardware actual y forzamos el mapeo de los pesos a este dispositivo
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return model_class.load_from_checkpoint(ckpt_path, map_location=target_device)

def extract_final_symmetries_cosine(pred_normals, pred_confs, pred_centers, conf_threshold, eps, min_samples):
    """
    Extrae las simetrías finales usando DBSCAN con métrica de Coseno precomputada.
    """
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
        
    # CORRECCIÓN VITAL: Distancia Coseno Absoluta
    dot_products = np.abs(np.dot(valid_normals, valid_normals.T))
    dot_products = np.clip(dot_products, 0.0, 1.0)
    distance_matrix = 1.0 - dot_products
    
    # Clustering con métrica precomputada
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
    # Reemplaza la línea anterior con tu lógica real de carga de modelo
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"⚡ Evaluando usando VRAM: {device}\n")
    
    # Configuración de los parámetros para el script oficial
    theta_cos = 1.0 - np.cos(np.radians(ANGLE_THRESHOLD)) 
    pdict = {
        "eps": EPSILON_RATE,
        "theta": theta_cos,  
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "rot_angle_threshold": 0.01
    }
    
    predictions_list = []

    print(f"📂 Abriendo base de datos HDF5: {TEST_H5_PATH}")
    with h5py.File(TEST_H5_PATH, 'r') as f:
        shape_ids = list(f.keys())
        
        with torch.no_grad():
            for shape_id in tqdm(shape_ids, desc="Generando Predicciones"):
                group = f[shape_id]
                points_np = group['points'][:]
                gt_planes = group['planar_symmetries'][:] if 'planar_symmetries' in group else []
                
                # Preprocesamiento (Asegúrate de tener t_points en la GPU y con shape [1, 3, N])
                t_points = torch.tensor(points_np, dtype=torch.float32)
                points_tensor = t_points.transpose(0, 1).unsqueeze(0).to(device)
                
                # Predicción Neuronal
                pred_n, pred_c, pred_cent = model(points_tensor)

                # NMS y Clustering corregido (¡Quitamos los [0]!)
                final_n, final_c, final_cent = extract_final_symmetries_cosine(
                    pred_n, pred_c, pred_cent, 
                    conf_threshold=CONFIDENCE_THRESHOLD,
                    eps=DBSCAN_EPS,
                    min_samples=DBSCAN_MIN_SAMPLES
                )
                
                # Formatear para el script del autor [Batch, M, 7]
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
                    
                predictions_list.append([
                    points_tensor.transpose(1, 2).cpu(), 
                    y_pred_tensor.unsqueeze(0).cpu(),    
                    [y_true_tensor.cpu()]                
                ])

    print("\n📊 Calculando métricas oficiales del paper...")
    total_map, total_phc, _ = calculate_metrics_from_predictions(
        predictions_list, 
        get_match_sequence_plane_symmetry, 
        pdict
    )
    
    map_val = total_map.item()
    phc_val = total_phc.item()

    print("="*50)
    print(f"🥇 Official Paper mAP: {map_val:.4f}")
    print(f"🥇 Official Paper PHC: {phc_val:.4f}")
    print("="*50)

    # ==========================================
    # GUARDAR RESULTADOS EN JSON
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
        }
    }

    # Ruta del archivo de log central (puedes cambiar el nombre o ruta si quieres)
    log_file = Path("eval_results_log.json")

    # Si el archivo ya existe, lo leemos para no borrar el historial
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = [] # Por si el archivo se corrompió
    else:
        logs = []

    # Añadimos el nuevo experimento y guardamos
    logs.append(experiment_data)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)

    print(f"💾 Resultados guardados exitosamente en {log_file.resolve()}")

if __name__ == "__main__":
    main()