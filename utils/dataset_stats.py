import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
# Apuntamos a la carpeta raíz del dataset crudo (Windows)
BASE_DIR = Path(r"C:\datasets\Symmetria-Easy-10k\sym-10k-xz-v1.0-repack-rotprob0.5-onlyrotx")
OUTPUT_DIR = Path(r"C:\datasets\Symmetria-Easy-10k-preproc") # Guardamos los gráficos aquí

def parse_sym_counts(filename):
    """Cuenta los tipos de simetría en un archivo .txt"""
    counts = {"plane": 0, "axis_discrete": 0, "axis_continuous": 0}
    
    if not filename.exists():
        return counts
        
    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines) <= 1:
            return counts
            
        line_amount = int(lines[0].strip())
        for line in lines[1:line_amount+1]:
            parts = line.strip().split()
            if not parts:
                continue
                
            if parts[0] == "plane":
                counts["plane"] += 1
            elif parts[0] == "axis":
                if parts[-1] == "inf":
                    counts["axis_continuous"] += 1
                else:
                    counts["axis_discrete"] += 1
    return counts

def main():
    splits = ['train', 'valid', 'test']
    
    # Lista para ir guardando la información de todos los modelos
    all_data = []

    for split in splits:
        split_dir = BASE_DIR / split
        if not split_dir.exists():
            print(f"⚠️ Carpeta no encontrada: {split_dir}")
            continue
            
        # Encontrar todas las subcarpetas (clases de figuras como 'astroid', 'citrus')
        classes = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for class_dir in tqdm(classes, desc=f"Procesando {split}"):
            class_name = class_dir.name
            
            # Buscar todos los txt en esta clase
            txt_files = list(class_dir.glob("*.txt"))
            
            for txt_path in txt_files:
                counts = parse_sym_counts(txt_path)
                
                # Guardamos un registro por cada archivo 3D
                all_data.append({
                    "split": split,
                    "class_name": class_name,
                    "planes": counts["plane"],
                    "discrete_axes": counts["axis_discrete"],
                    "continuous_axes": counts["axis_continuous"]
                })

    # Crear un DataFrame con pandas
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("❌ No se encontraron datos para procesar.")
        return

    # 1. Agrupar por clase y sacar el PROMEDIO de simetrías por figura
    stats_df = df.groupby('class_name')[['planes', 'discrete_axes', 'continuous_axes']].mean().reset_index()
    
    # Redondear para que sea más legible
    stats_df = stats_df.round(2)
    
    # Guardar en CSV
    csv_path = OUTPUT_DIR / "dataset_symmetries_stats.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"\n✅ ¡CSV guardado en {csv_path}!")

    # 2. Generar un gráfico de barras apiladas
    stats_df.set_index('class_name').plot(
        kind='bar', 
        stacked=True, 
        figsize=(12, 6),
        colormap='viridis'
    )
    plt.title("Promedio de Simetrías por Clase (Easy-10k)")
    plt.ylabel("Cantidad de Simetrías")
    plt.xlabel("Clase (Figura)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Guardar el gráfico
    plot_path = OUTPUT_DIR / "dataset_symmetries_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✅ ¡Gráfico guardado en {plot_path}!\n")

if __name__ == "__main__":
    main()