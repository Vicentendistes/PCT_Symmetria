import os
import json
import yaml
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Cambia esta ruta a la carpeta donde tienes guardados tus JSON actuales
JSON_DIR = "resultados_evaluacion/Symmetria-Hard-10k-preproc" 

def update_jsons_with_hparams(directory_path):
    dir_path = Path(directory_path)
    
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"❌ La carpeta {dir_path} no existe o no es un directorio válido.")
        return

    json_files = list(dir_path.glob("*.json"))
    
    if not json_files:
        print(f"⚠️ No se encontraron archivos .json en {dir_path}")
        return

    print(f"🔍 Encontrados {len(json_files)} archivos JSON. Iniciando actualización...\n")

    actualizados = 0
    errores = 0

    for json_file in json_files:
        print(f"Procesando: {json_file.name}")
        
        try:
            # 1. Leer el JSON actual
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Si ya tiene hparams, lo saltamos para no sobreescribir innecesariamente
            if "hparams" in data:
                print("  -> ⏩ Ya contiene 'hparams'. Saltando.")
                continue
            
            # 2. Extraer la ruta del checkpoint
            ckpt_path_str = data.get("model_checkpoint")
            if not ckpt_path_str:
                print("  -> ⚠️ No se encontró 'model_checkpoint' en el JSON. Saltando.")
                errores += 1
                continue
            
            # 3. Calcular la ruta del hparams.yaml (dos niveles arriba del .ckpt)
            ckpt_path = Path(ckpt_path_str)
            hparams_path = ckpt_path.parent.parent / "hparams.yaml"
            
            hparams_data = {}
            if hparams_path.exists():
                with open(hparams_path, 'r', encoding='utf-8') as hf:
                    hparams_data = yaml.safe_load(hf)
                print(f"  -> ✅ hparams.yaml encontrado y cargado.")
            else:
                print(f"  -> ⚠️ hparams.yaml NO encontrado en {hparams_path}. Se insertará vacío.")

            # 4. Reconstruir el diccionario para que 'hparams' quede justo debajo de 'model_checkpoint'
            new_data = {}
            for key, value in data.items():
                new_data[key] = value
                if key == "model_checkpoint":
                    new_data["hparams"] = hparams_data

            # 5. Sobreescribir el JSON con la nueva estructura
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
                
            actualizados += 1
            
        except Exception as e:
            print(f"  -> ❌ Error procesando el archivo: {e}")
            errores += 1

    print("\n" + "="*50)
    print("🎯 RESUMEN DE ACTUALIZACIÓN")
    print("="*50)
    print(f"Archivos procesados exitosamente: {actualizados}")
    print(f"Archivos con errores o sin checkpoint: {errores}")
    print("="*50)

if __name__ == "__main__":
    update_jsons_with_hparams(JSON_DIR)