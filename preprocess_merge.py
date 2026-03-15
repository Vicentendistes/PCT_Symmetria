import h5py
import lzma
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

from src.dataset.transforms.UnitSphereNormalization import UnitSphereNormalization

# ==========================================
# CONFIGURACIÓN DE RUTAS MAESTRAS
# ==========================================
# Directorio base donde están tus carpetas descomprimidas
BASE_DATA_DIR = Path("/data/vimunoz")

# Lista de datasets que queremos fusionar (Excluimos Hard y SSL)
DATASETS_TO_MERGE = [
    "Intermediate-2-10k/sym-10k-xz-v3.0-9classes-rotprob0.75-rotxroty"  # Revisa si el nombre de la subcarpeta es exacto
]

# Dónde dejaremos los 3 archivos HDF5 unificados
OUTPUT_DIR = Path("/home/vimunoz/proyectos/Symmetria-Master-10k-hdf5")
NUM_POINTS_FPS = 1024

def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, bool]:
    num_points = points.shape[0]
    if num_points <= num_samples:
        return points, True 
        
    centroids = torch.zeros(num_samples, dtype=torch.long)
    distance = torch.ones(num_points) * 1e10
    farthest = torch.randint(0, num_points, (1,), dtype=torch.long)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :].view(1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return points[centroids], False

def parse_sym_file_numpy(filename):
    planar, axis_cont, axis_disc = [], [], []
    if not filename.exists():
        return np.array(planar), np.array(axis_cont), np.array(axis_disc)
        
    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            return np.array(planar), np.array(axis_cont), np.array(axis_disc)
            
        line_amount = int(lines[0].strip())
        for line in lines[1:line_amount+1]:
            parts = line.strip().split()
            if parts[0] == "plane":
                planar.append([float(x) for x in parts[1:]])
            elif parts[0] == "axis" and parts[-1] == "inf":
                axis_cont.append([float(x) for x in parts[1:7]])
            else:
                axis_disc.append([float(x) for x in parts[1:]])
                
    return (np.array(planar, dtype=np.float32), 
            np.array(axis_cont, dtype=np.float32), 
            np.array(axis_disc, dtype=np.float32))

def main():
    splits = ['train', 'valid', 'test']
    normalizador = UnitSphereNormalization(eps=1.0e-9)
    
    # Limpiamos el directorio de salida si ya existía para evitar sobreescribir mal
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for split in splits:
        out_file = OUTPUT_DIR / f"{split}.h5"
        print(f"\n📦 Generando archivo maestro: {out_file.name}")
        
        # Abrimos el HDF5 en modo 'a' (append) para ir agregándole datos de cada carpeta
        with h5py.File(out_file, 'a') as h5f:
            
            for dataset_rel_path in DATASETS_TO_MERGE:
                # Nombre corto para identificar de dónde vino (ej: "Easy-10k")
                dataset_name = dataset_rel_path.split('/')[0] 
                split_dir = BASE_DATA_DIR / dataset_rel_path / split
                
                xz_files = list(split_dir.rglob("*.xz"))
                if not xz_files:
                    print(f"⚠️ No se encontraron archivos en {split_dir}")
                    continue
                
                fps_evitados = 0
                
                for xz_path in tqdm(xz_files, desc=f"Procesando {dataset_name} ({split})"):
                    # Prefijo para evitar colisiones de IDs entre carpetas
                    shape_id = f"{dataset_name}_{xz_path.stem}" 
                    
                    with lzma.open(xz_path, 'rb') as f:
                        points = np.loadtxt(f)
                    
                    sym_path = xz_path.with_name(xz_path.name.replace('.xz', '-sym.txt'))
                    p_sym, ac_sym, ad_sym = parse_sym_file_numpy(sym_path)
                    
                    t_points = torch.tensor(points, dtype=torch.float32)
                    t_psym = torch.tensor(p_sym, dtype=torch.float32) if len(p_sym) > 0 else None
                    t_acsym = torch.tensor(ac_sym, dtype=torch.float32) if len(ac_sym) > 0 else None
                    t_adsym = torch.tensor(ad_sym, dtype=torch.float32) if len(ad_sym) > 0 else None
                    
                    t_points, ya_aplicado = farthest_point_sampling(t_points, NUM_POINTS_FPS)
                    if ya_aplicado:
                        fps_evitados += 1
                    
                    _, norm_points, norm_psym, norm_acsym, norm_adsym = normalizador(
                        0, t_points, t_psym, t_acsym, t_adsym
                    )
                    
                    grp = h5f.create_group(shape_id)
                    grp.create_dataset('points', data=norm_points.numpy())
                    grp.create_dataset('planar_symmetries', data=norm_psym.numpy() if norm_psym is not None else np.array([]))
                    # Mantenemos los datasets de ejes vacíos o con datos por compatibilidad, aunque no los usemos en el Loss
                    grp.create_dataset('axis_continue_symmetries', data=norm_acsym.numpy() if norm_acsym is not None else np.array([]))
                    grp.create_dataset('axis_discrete_symmetries', data=norm_adsym.numpy() if norm_adsym is not None else np.array([]))

                if fps_evitados > 0:
                    print(f"👉 {fps_evitados} modelos en {dataset_name} omitieron FPS.")

if __name__ == "__main__":
    main()