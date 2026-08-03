import h5py
import lzma
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ¡Importamos tu transform directamente!
from src.dataset.transforms.UnitSphereNormalization import UnitSphereNormalization

# ==========================================
# CONFIGURACIÓN DE RUTAS (¡Revisa esto antes de correr!)
# ==========================================
# Tu dataset (asegúrate de que apunte a la carpeta que contiene train, valid y test)
BASE_DIR = Path("C:\\Users\\vicho\\Escritorio\\Tesis II\\symmetry-detection-pct\\data\\ShapeNet_Symmetry_850\\todo")
OUTPUT_DIR = Path("C:\\Users\\vicho\\Escritorio\\Tesis II\\symmetry-detection-pct\\data\\ShapeNet_Symmetry_850-preproc")
NUM_POINTS_FPS = 1024
FPS_SEED = None

def farthest_point_sampling(
    points: torch.Tensor,
    num_samples: int,
    generator: torch.Generator | None = None
) -> Tuple[torch.Tensor, bool]:
    """
    Aplica FPS a una nube de puntos [N, 3] para reducirla a num_samples.
    Retorna los puntos muestreados y un booleano 'ya_aplicado' para avisar.
    """
    num_points = points.shape[0]
    
    # Si la nube ya tiene 1024 puntos o menos, retornamos directamente
    if num_points <= num_samples:
        return points, True 
        
    centroids = torch.zeros(num_samples, dtype=torch.long)
    distance = torch.ones(num_points) * 1e10
    farthest = torch.randint(0, num_points, (1,), dtype=torch.long, generator=generator)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :].view(1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return points[centroids], False

def parse_sym_file_numpy(filename):
    """Extrae las simetrías crudas de los .txt"""
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
    normalizador = UnitSphereNormalization(eps=1.0e-9)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    xz_files = sorted(BASE_DIR.rglob("*.xz"))
    
    if not xz_files:
        print(f"⚠️ Atención: No se encontraron archivos .xz en {BASE_DIR}")
        return
        
    out_file = OUTPUT_DIR / f"shapenet.h5"
    print(f"\n📦 Empaquetando {len(xz_files)} modelos en {out_file.name}...")
    
    fps_evitados = 0  # Contador inteligente para no romper la consola
    
    fps_generator = None
    if FPS_SEED is not None:
        fps_generator = torch.Generator().manual_seed(FPS_SEED)

    with h5py.File(out_file, 'w') as h5f:
        for xz_path in tqdm(xz_files, desc=f"Procesando"):
            shape_id = xz_path.stem 
            
            # 1. Leer Puntos Crudos (LZMA)
            with lzma.open(xz_path, 'rb') as f:
                points = np.loadtxt(f)
            
            # 2. Extraer Simetrías
            sym_path = xz_path.with_name(xz_path.name.replace('.xz', '-sym.txt'))
            p_sym, ac_sym, ad_sym = parse_sym_file_numpy(sym_path)
            
            # Convertir a Tensores de PyTorch para las matemáticas
            t_points = torch.tensor(points, dtype=torch.float32)
            t_psym = torch.tensor(p_sym, dtype=torch.float32) if len(p_sym) > 0 else None
            t_acsym = torch.tensor(ac_sym, dtype=torch.float32) if len(ac_sym) > 0 else None
            t_adsym = torch.tensor(ad_sym, dtype=torch.float32) if len(ad_sym) > 0 else None
            
            # 3. Aplicar FPS (Y chequear si ya estaba listo)
            t_points, ya_aplicado = farthest_point_sampling(t_points, NUM_POINTS_FPS, generator=fps_generator)
            if ya_aplicado:
                fps_evitados += 1
            
            # 4. Aplicar Normalización de la Esfera Unitaria
            _, norm_points, norm_psym, norm_acsym, norm_adsym = normalizador(
                0, t_points, t_psym, t_acsym, t_adsym
            )
            
            # 5. Guardar en HDF5 (Nativo Numpy para máxima velocidad de lectura)
            grp = h5f.create_group(shape_id)
            grp.create_dataset('points', data=norm_points.numpy())
            grp.create_dataset('planar_symmetries', data=norm_psym.numpy() if norm_psym is not None else np.array([]))
            grp.create_dataset('axis_continue_symmetries', data=norm_acsym.numpy() if norm_acsym is not None else np.array([]))
            grp.create_dataset('axis_discrete_symmetries', data=norm_adsym.numpy() if norm_adsym is not None else np.array([]))
    
    # Resumen de FPS para esa carpeta
    if fps_evitados > 0:
        print(f"👉 Aviso: {fps_evitados} modelos ya tenían {NUM_POINTS_FPS} puntos o menos (Se omitió el FPS).")

if __name__ == "__main__":
    main()
