# PointNet global de Santelices

## Que hace el YAML descargado

El archivo original configura un modelo global, no una prediccion densa:

1. PointNet transforma la nube completa en un vector de 1024 componentes.
2. Una cabeza compartida predice un unico centro para la figura.
3. Cada una de las 32 cabezas independientes predice una normal y una
   confianza.
4. Cada plano se representa como
   `[nx, ny, nz, px, py, pz, confidence]`.
5. La asignacion entre cabezas y planos reales usa el costo angular
   `1 - |n_pred dot n_gt|`.

Con 32 cabezas y sin BatchNorm, esta arquitectura tiene aproximadamente
24,25 millones de parametros entrenables. La mayor parte no esta en PointNet,
sino en las 32 MLP independientes de prediccion. Por eso no conviene asumir
que sera una corrida liviana solo por usar un encoder mas simple.

Los ejes rotacionales estan desactivados porque sus cantidades son cero.
`w1`, `w2` y `w3` solo combinaban las tres familias de simetria; con planos
solamente, `w1=1` era el unico valor activo.

El optimizador original es Adam con sus valores por defecto: `lr=0.001` y
sin `weight_decay`. No hay scheduler. El YAML guardaba todos los checkpoints
y entrenaba por 10 epocas con `batch_size=1`.

`theta=0.00015230484` no esta expresado en grados ni radianes. Es
aproximadamente `1-cos(1 grado)`, que coincide con la distancia usada por
`SymPlane`.

Hay dos parametros del YAML cuyo efecto real no coincide completamente con
su nombre:

- `ReflectionSymmetryDistance(p=1)` conserva el argumento, pero la
  implementacion usada llama internamente a `get_sde` sin reenviar `p`; por
  tanto, ejecuta la norma por defecto `p=2`.
- `NormalLoss(reduction=...)` normaliza las normales y calcula directamente
  una media; el valor de `reduction` no cambia ese `forward`.

La adaptacion conserva ese comportamiento para reproducir el baseline. No se
deben "corregir" esos detalles dentro de la misma corrida, porque se estaria
evaluando otra funcion de perdida.

## Adaptacion a HDF5

`SymDataset` ya lee `points` y `planar_symmetries` desde HDF5 y
`SymDatasetBatcher` conserva una lista de planos por figura. Por eso no hace
falta convertir el dataset a otro formato.

La configuracion adaptada usa `IdentityTransform` porque los archivos
`*-preproc` ya contienen FPS y normalizacion a esfera unitaria. Aplicar de
nuevo `UnitSphereNormalization` produciria una segunda normalizacion.

El campo `n_points=8192` del YAML descargado no interviene en el `forward` de
PointNet. En esta adaptacion, la cantidad efectiva es la almacenada en HDF5
(1024 puntos).

El modelo descargado arma las cabezas con
`vstack(...).view(B, M, 4)`. Eso solo mantiene el orden correcto cuando
`B=1`. `SantelicesPointNetGlobal` usa `stack(..., dim=1)`, da el mismo
resultado para `B=1` y permite probar lotes mayores sin mezclar ejemplos.

## Ejecucion recomendada

Primero verificar imports, HDF5, perdida y un paso de optimizacion:

```bash
python main_pointnet.py fit \
  --config configs/pointnet/hdf5/hard100k.yaml \
  --trainer.fast_dev_run=true
```

Luego hacer una prueba breve sobre Hard-10k:

```bash
python main_pointnet.py fit \
  --config configs/pointnet/hdf5/hard100k.yaml \
  --data.dataset_path=/data/vimunoz/Symmetria-Hard-10k-preproc \
  --trainer.max_epochs=1 \
  --trainer.limit_train_batches=200 \
  --trainer.limit_val_batches=50
```

Si ambas pruebas terminan sin errores, ejecutar el protocolo adaptado:

```bash
python main_pointnet.py fit \
  --config configs/pointnet/hdf5/hard100k.yaml
```

La evaluacion del baseline global no usa DBSCAN. Sus 32 salidas ya son
candidatos por figura y se ordenan por confianza:

```bash
python -m src.metrics.official_eval_pointnet \
  --checkpoint logs/pointnet-global-santelices-hard100k/version_0/checkpoints/last.ckpt \
  --test-h5 /data/vimunoz/Symmetria-Hard-10k-preproc/test.h5
```
