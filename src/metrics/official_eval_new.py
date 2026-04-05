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
from torch.utils.data import Dataset, DataLoader
import concurrent.futures

# Asegúrate de que estas rutas coincidan con tu proyecto
from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry

# ==========================================
# CONFIGURACIÓN DE PARÁMETROS
# ==========================================

TEST_H5_PATH = "/data/vimunoz/Symmetria-Intermediate-2-100k-preproc/test.h5"
CHECKPOINT_PATH = "/home/vimunoz/proyectos/PCT_Symmetria/logs/intermediate-2-100k-64-MHA-Optimized/version_2/checkpoints/last.ckpt"
MODEL_CLASS_PATH = "src.model.LightningSymmetryModel.LightningSymmetryModel"

CONFIDENCE_THRESHOLD = 0.1
ANGLE_THRESHOLD = 1.0       # Grados permitidos de error
EPSILON_RATE = 0.01         # Porcentaje de la diagonal para el error de distancia
DBSCAN_EPS = 0.005          # Umbral de distancia Coseno (aprox 5 grados)
DBSCAN_MIN_SAMPLES = 10
BATCH_SIZE = 16           # <-- NUEVO: Procesaremos 16 figuras a la vez
NUM_WORKERS = 8             # <-- NUEVO: Hilos de CPU para cargar datos

# ==========================================
# CLASES DE DATOS (NUEVO)
# ==========================================

# ---------------------------------------------------------
# 1. NUEVA FUNCIÓN WORKER (Ponla FUERA de main() para mayor limpieza)
# ---------------------------------------------------------
def dbscan_worker(args):
    """Ejecuta un DBSCAN individual de forma aislada para poder paralelizar"""
    if args is None:
        return np.array([]), np.array([]), np.array([])
        
    eps, min_samples, distance_matrix_np, valid_normals_np, valid_confs_np, valid_centers_np = args
    
    # Clustering en CPU
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix_np)
    labels = clustering.labels_
    
    final_normals, final_confs, final_centers = [], [], []
    for label in set(labels):
        if label == -1: continue 
            
        cluster_mask = (labels == label)
        cluster_normals = valid_normals_np[cluster_mask]
        cluster_confs = valid_confs_np[cluster_mask]
        cluster_centers = valid_centers_np[cluster_mask]
        
        best_idx = np.argmax(cluster_confs)
        final_normals.append(cluster_normals[best_idx])
        final_confs.append(cluster_confs[best_idx])
        final_centers.append(cluster_centers[best_idx])
        
    return np.array(final_normals), np.array(final_confs), np.array(final_centers)




class SymmetriaDataset(Dataset):
    """Dataset optimizado para no ahogar el I/O del disco"""
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # Leemos las llaves una sola vez al inicio
        with h5py.File(h5_path, 'r') as f:
            self.shape_ids = list(f.keys())
        self.file = None # El archivo se abrirá luego

    def __len__(self):
        return len(self.shape_ids)

    def __getitem__(self, idx):
        # Magia aquí: Solo abrimos el archivo H5 UNA vez por cada Worker
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            
        shape_id = self.shape_ids[idx]
        group = self.file[shape_id]
        
        # Leemos directo a memoria
        points_np = group['points'][:]
        gt_planes = group['planar_symmetries'][:] if 'planar_symmetries' in group else np.empty((0, 6))

        points_tensor = torch.tensor(points_np, dtype=torch.float32).transpose(0, 1)
        gt_tensor = torch.tensor(gt_planes, dtype=torch.float32)

        return shape_id, points_tensor, gt_tensor
    
    def __del__(self):
        # Aseguramos cerrar el archivo al terminar
        if self.file is not None:
            self.file.close()
def custom_collate_fn(batch):
    """
    Agrupa los datos en batches. Necesario porque 'gt_tensor' 
    puede tener diferente número de simetrías por figura.
    """
    shape_ids = [item[0] for item in batch]
    # Apilamos los puntos normalmente: (B, 3, N)
    points_batch = torch.stack([item[1] for item in batch])
    # Dejamos los Ground Truths en una lista porque sus tamaños varían
    gts_list = [item[2] for item in batch]
    
    return shape_ids, points_batch, gts_list

# ==========================================
# FUNCIONES DEL MODELO Y PROCESAMIENTO
# ==========================================

def load_model_dynamically(class_path, ckpt_path):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model_class.load_from_checkpoint(ckpt_path, map_location=target_device)

def extract_final_symmetries_cosine_optimized(pred_normals, pred_confs, pred_centers, conf_threshold, eps, min_samples):
    """Versión optimizada: Hace todo el filtrado y matemáticas pesadas en la GPU."""
    M = pred_normals.shape[2] 
    
    # 1. Operaciones en GPU: Reformateo
    normals_flat = pred_normals.reshape(-1, 3)
    confs_flat = pred_confs.reshape(-1)
    
    centers_expanded = pred_centers.unsqueeze(2).expand(-1, -1, M, -1)
    centers_flat = centers_expanded.reshape(-1, 3)
    
    # 2. Filtrado en GPU
    mask = confs_flat > conf_threshold
    valid_normals = normals_flat[mask]
    valid_confs = confs_flat[mask]
    valid_centers = centers_flat[mask]
    
    if len(valid_normals) == 0:
        return np.array([]), np.array([]), np.array([])
        
    # 3. Cálculo de Distancia del Coseno en GPU (Esto es lo que acelera todo)
    dot_products = torch.abs(torch.mm(valid_normals, valid_normals.t()))
    dot_products = torch.clamp(dot_products, 0.0, 1.0)
    distance_matrix = 1.0 - dot_products
    
    # 4. Enviar a CPU SOLO para DBSCAN
    distance_matrix_np = distance_matrix.cpu().numpy()
    valid_normals_np = valid_normals.cpu().numpy()
    valid_confs_np = valid_confs.cpu().numpy()
    valid_centers_np = valid_centers.cpu().numpy()
    
    # 5. Clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix_np)
    labels = clustering.labels_
    
    final_normals, final_confs, final_centers = [], [], []
    for label in set(labels):
        if label == -1: continue 
            
        cluster_mask = (labels == label)
        cluster_normals = valid_normals_np[cluster_mask]
        cluster_confs = valid_confs_np[cluster_mask]
        cluster_centers = valid_centers_np[cluster_mask]
        
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
    print(f"⚡ Evaluando usando VRAM: {device} | Batch Size: {BATCH_SIZE}\n")
    
    theta_cos = 1.0 - np.cos(np.radians(ANGLE_THRESHOLD)) 
    pdict = {
        "eps": EPSILON_RATE,
        "theta": theta_cos,  
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "rot_angle_threshold": 0.01
    }
    
    predictions_list = []
    predictions_by_category = defaultdict(list) 

    # --- NUEVO: PREPARACIÓN DEL DATALOADER ---
    print(f"📂 Preparando Dataloader desde: {TEST_H5_PATH}")
    dataset = SymmetriaDataset(TEST_H5_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=NUM_WORKERS,
        pin_memory=True # Acelera la transferencia CPU -> GPU
    )

    # ---------------------------------------------------------
# 2. REEMPLAZA EL BUCLE INTERNO DENTRO DE main()
# ---------------------------------------------------------
    with torch.no_grad():
        for shape_ids, points_batch, gts_list in tqdm(dataloader, desc="Procesando Batches"):
            
            points_batch = points_batch.to(device)
            
            # --- A. Inferencia ultrarrápida en GPU ---
            pred_n_batch, pred_c_batch, pred_cent_batch = model(points_batch)

            batch_args = []
            
            # --- B. Preparar las matrices en GPU y pasarlas a CPU ---
            for i in range(len(shape_ids)):
                pred_n = pred_n_batch[i:i+1]
                pred_c = pred_c_batch[i:i+1]
                pred_cent = pred_cent_batch[i:i+1]
                
                M = pred_n.shape[2] 
                normals_flat = pred_n.reshape(-1, 3)
                confs_flat = pred_c.reshape(-1)
                
                centers_expanded = pred_cent.unsqueeze(2).expand(-1, -1, M, -1)
                centers_flat = centers_expanded.reshape(-1, 3)
                
                mask = confs_flat > CONFIDENCE_THRESHOLD
                valid_normals = normals_flat[mask]
                valid_confs = confs_flat[mask]
                valid_centers = centers_flat[mask]
                
                if len(valid_normals) == 0:
                    batch_args.append(None)
                    continue
                
                # Multiplicación matricial sigue en GPU (muy rápido)
                dot_products = torch.abs(torch.mm(valid_normals, valid_normals.t()))
                dot_products = torch.clamp(dot_products, 0.0, 1.0)
                distance_matrix = 1.0 - dot_products
                
                # Empaquetamos los tensores ya en formato NumPy para el multihilo
                batch_args.append((
                    DBSCAN_EPS,
                    DBSCAN_MIN_SAMPLES,
                    distance_matrix.cpu().numpy(),
                    valid_normals.cpu().numpy(),
                    valid_confs.cpu().numpy(),
                    valid_centers.cpu().numpy()
                ))

            # --- C. ¡LA MAGIA! Ejecutar los 16 DBSCAN en paralelo en la CPU ---
            with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                # Map ejecuta todos los DBSCAN simultáneamente manteniendo el orden original
                results = list(executor.map(dbscan_worker, batch_args))
                
            # --- D. Empaquetar y Guardar Resultados ---
            for i in range(len(shape_ids)):
                shape_id = shape_ids[i]
                gt_tensor = gts_list[i]
                final_n, final_c, final_cent = results[i] # Obtenemos el resultado paralelo
                
                if len(final_n) > 0:
                    y_pred_tensor = torch.cat([
                        torch.tensor(final_n, dtype=torch.float32),
                        torch.tensor(final_cent, dtype=torch.float32),
                        torch.tensor(final_c, dtype=torch.float32).unsqueeze(1)
                    ], dim=1)
                else:
                    y_pred_tensor = torch.empty((0, 7), dtype=torch.float32)

                single_points_cpu = points_batch[i].unsqueeze(0).transpose(1, 2).cpu()

                prediction_item = [
                    single_points_cpu, 
                    y_pred_tensor.unsqueeze(0),    
                    [gt_tensor]                
                ]
                
                predictions_list.append(prediction_item)
                
                try:
                    categoria = shape_id.split('-')[1]
                except IndexError:
                    categoria = "desconocido" 
                
                predictions_by_category[categoria].append(prediction_item)

    # --- MÉTRICAS GLOBALES ---
    print("\n📊 Calculando métricas oficiales globales...")
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
            "DBSCAN_MIN_SAMPLES": DBSCAN_MIN_SAMPLES,
            "BATCH_SIZE": BATCH_SIZE
        },
        "official_metrics": {
            "mAP": round(map_val, 4),
            "PHC": round(phc_val, 4)
        },
        "metrics_by_category": category_metrics_log
    }

    log_file = Path("eval_results_log.json")

    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = [] 
    else:
        logs = []

    logs.append(experiment_data)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)

    print(f"💾 Resultados guardados exitosamente en {log_file.resolve()}")

if __name__ == "__main__":
    main()