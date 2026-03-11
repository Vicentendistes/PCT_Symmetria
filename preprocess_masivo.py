import os
import lzma
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- RUTAS EN EL SERVIDOR  ---
INPUT_ROOT = Path("/data/vimunoz/raw_datasets")
OUTPUT_ROOT = Path("/data/vimunoz/symmetria-fps-1024")
N_SAMPLES = 1024

def farthest_point_sampling(coords_xyz: np.ndarray, n_samples: int) -> np.ndarray:
    N = coords_xyz.shape[0]
    if n_samples >= N:
        return np.arange(N)
    centroids = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float64)
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = coords_xyz[farthest, :].reshape(1, 3)
        diff = coords_xyz - centroid
        dist = np.einsum("ij,ij->i", diff, diff)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = int(np.argmax(distances))
    return centroids

def process_file(in_path: Path):
    try:
        rel = in_path.relative_to(INPUT_ROOT)
        out_path = OUTPUT_ROOT / rel

        # Si el archivo ya existe (ej. el Easy-10k), lo salta para no perder tiempo
        if out_path.exists():
            return True

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with lzma.open(in_path, "rb") as f:
            points = np.loadtxt(f)
        if points.ndim == 1:
            points = points[None, :]

        idx = farthest_point_sampling(points[:, :3], N_SAMPLES)
        points_sub = points[idx, :]

        with lzma.open(out_path, "wb") as f:
            np.savetxt(f, points_sub, fmt="%.8f")

        # Copiar archivo de simetría
        sym_filename = in_path.name.replace('.xz', '-sym.txt')
        sym_in_path = in_path.with_name(sym_filename)
        sym_out_path = out_path.with_name(sym_filename)

        if sym_in_path.exists():
            shutil.copy(sym_in_path, sym_out_path)

        return True
    except Exception as e:
        print(f"Error procesando {in_path}: {e}")
        return False

def main():
    print(f"Buscando archivos en {INPUT_ROOT}...")
    xz_files = list(INPUT_ROOT.rglob("*.xz"))
    print(f"Encontrados {len(xz_files)} archivos .xz para procesar.")
    
    # Usamos 14 workers paralelos (el Ryzen tiene 16 hilos)
    with ProcessPoolExecutor(max_workers=14) as executor:
        list(tqdm(executor.map(process_file, xz_files), total=len(xz_files)))
        
    print("¡Procesamiento masivo completado!")

if __name__ == "__main__":
    main()