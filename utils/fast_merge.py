import h5py
import os
from tqdm import tqdm

def merge_preprocessed_hdf5_groups():
    base_path = "/home/vimunoz/data"
    input_dirs = [
        os.path.join(base_path, "Symmetria-Easy-100k-preproc"),
        os.path.join(base_path, "Symmetria-Intermediate-1-100k-preproc"),
        os.path.join(base_path, "Symmetria-Intermediate-2-100k-preproc")
    ]
    
    output_dir = os.path.join(base_path, "Symmetria-Merge-300k-preproc")
    os.makedirs(output_dir, exist_ok=True)
    splits = ["train.h5", "valid.h5", "test.h5"]
    
    for split in splits:
        out_file_path = os.path.join(output_dir, split)
        print(f"\n📦 Generando archivo maestro: {split}")
        
        with h5py.File(out_file_path, 'w') as out_f:
            for in_dir in input_dirs:
                file_path = os.path.join(in_dir, split)
                dataset_name = os.path.basename(in_dir).replace('-preproc', '')
                
                if not os.path.exists(file_path):
                    print(f"⚠️ Archivo no encontrado: {file_path}")
                    continue
                    
                with h5py.File(file_path, 'r') as in_f:
                    # Copiamos cada grupo (forma 3D) al nuevo HDF5
                    for shape_id in tqdm(in_f.keys(), desc=f"Copiando {dataset_name}"):
                        # Añadimos un prefijo por seguridad para evitar colisiones de IDs
                        new_shape_id = f"{dataset_name}_{shape_id}" if not shape_id.startswith(dataset_name) else shape_id
                        in_f.copy(shape_id, out_f, name=new_shape_id)

if __name__ == "__main__":
    merge_preprocessed_hdf5_groups()