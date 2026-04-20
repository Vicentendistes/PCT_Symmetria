# 🛸 PROYECTO MASTER: Detección de Simetría 3D con PCT

Este proyecto utiliza PyTorch y PyTorch Lightning para la detección densa de planos de simetría usando la arquitectura PCT_M1.

## Despliegue en Servidores RELELA (Desde Cero)

### 1. Requisito Previo: Instalación de Miniconda
Si al entrar al servidor ejecutas `conda` y el comando no existe, instala Miniconda directamente desde la terminal:

```bash
# Descargar el instalador oficial
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Ejecutar la instalación
bash Miniconda3-latest-Linux-x86_64.sh

# INSTRUCCIONES DURANTE LA INSTALACIÓN:
# 1. Presiona Enter para leer la licencia (puedes presionar 'q' para saltar al final).
# 2. Escribe 'yes' para aceptar los términos.
# 3. Presiona Enter para aceptar la ruta por defecto (usualmente /home/tu_usuario/miniconda3).
# 4. MUY IMPORTANTE: Escribe 'yes' cuando pregunte si deseas inicializar Conda (conda init).

# Recargar la configuración para activar Conda sin tener que salir del servidor
source ~/.bashrc
```

### 2. Transferencia del Proyecto
* **Vía Git:** `git clone https://github.com/Vicentendistes/PCT_Symmetria.git` y luego `cd PCT_Symmetria`
* **Vía WinSCP:** Arrastra tu carpeta local `PCT_Symmetria` al servidor y entra a ella.

### 3. Configuración del Entorno Virtual (Aislado)
Para evitar corromper el entorno base del servidor, crearemos un entorno llamado `symclean` usando Conda para la gestión base y Pip para el control de paquetes.

```bash
# 1. Crear el entorno con Python 3.10 usando el archivo base
conda env create -f environment.yml

# 2. Activar el entorno
conda activate symclean


# 3. Instalar absolutamente todo (PyTorch con CUDA incluido) de una sola vez
pip install -r requirements.txt
```

### 4. Ejecución Segura y Sanity Check
Siempre usa `tmux` para evitar que el proceso muera si se corta tu conexión SSH o el túnel.

```bash
# Iniciar una sesión segura
tmux new -s entrenamiento_m1

# Asegúrate de activar el entorno dentro de tmux
conda activate symclean

# Ejecutar prueba de vida (1 solo batch) para confirmar GPUs y Tensores
python main.py --config configs/easy/pct_m1.yaml --fast_dev_run True
```
*(Nota: Para salir de tmux dejándolo correr de fondo presiona `Ctrl+B` y luego `D`. Para volver a entrar usa `tmux attach -t entrenamiento_m1`).*

### 4. Entrenamiento
```bash
python main.py fit --config <PATH/TO/CONFIG>
```

### 5. Evaluación
```bash
python -m src.metrics.official_eval
```