# src/dataset/transforms/RandomRotation3D.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
from scipy.spatial.transform import Rotation

from src.dataset.transforms.AbstractTransform import AbstractTransform


class RandomRotation(AbstractTransform):
    """
    Aplica una rotación 3D aleatoria con distribución uniforme SO(3) a la nube de puntos 
    y a sus simetrías (vectores normales y puntos de anclaje).
    
    Versión Stateless: No guarda caché interno de matrices. Optimizada para no saturar 
    la VRAM y ser 100% segura al usar num_workers > 0 en DataLoaders paralelos.
    """

    def __init__(self) -> None:
        # Al ser stateless, ya no necesitamos inicializar diccionarios de caché ni 'keep_cache'
        pass

    # ---------- helpers internos ----------

    def _get_random_rotation_matrix(self) -> torch.Tensor:
        """Genera una matriz de rotación 3D aleatoria con distribución uniforme SO(3)."""
        R_numpy = Rotation.random().as_matrix()
        return torch.tensor(R_numpy, dtype=torch.float32)

    def _rotate_points(self, points: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # Multiplicación matricial: (N, 3) @ (3, 3) = (N, 3)
        return points @ R.T

    def _rotate_syms(self, syms: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # Validación de seguridad para evitar errores silenciosos
        assert syms.shape[1] >= 6, "El tensor de simetrías debe tener al menos 6 columnas (3 normales, 3 anclajes)"
        
        syms_rotated = syms.clone()
        # Rotar las normales (columnas 0:3)
        syms_rotated[:, 0:3] = syms[:, 0:3] @ R.T
        # Rotar los centros/puntos de anclaje (columnas 3:6)
        syms_rotated[:, 3:6] = syms[:, 3:6] @ R.T
        return syms_rotated

    # ---------- API AbstractTransform ----------

    def transform(
        self,
        idx: int,
        points: torch.Tensor,
        planar_symmetries: Optional[torch.Tensor],
        axis_continue_symmetries: Optional[torch.Tensor],
        axis_discrete_symmetries: Optional[torch.Tensor],
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # 1. Generar la matriz de rotación y enviarla al device de la nube de puntos
        R = self._get_random_rotation_matrix().to(points.device)

        # 2. Rotar la nube de puntos
        points = self._rotate_points(points, R)

        # 3. Rotar los Ground Truths correspondientes
        if planar_symmetries is not None:
            planar_symmetries = self._rotate_syms(planar_symmetries, R)
        if axis_continue_symmetries is not None:
            axis_continue_symmetries = self._rotate_syms(axis_continue_symmetries, R)
        if axis_discrete_symmetries is not None:
            axis_discrete_symmetries = self._rotate_syms(axis_discrete_symmetries, R)

        # Retornamos todo directamente sin guardar el estado local
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    # ---------- Métodos Heredados Obsoletos ----------
    
    def inverse_transform(self, *args, **kwargs):
        """Bloqueado intencionalmente: Esta transformación ahora es stateless."""
        raise NotImplementedError("RandomRotation3D ahora es stateless. No se guarda caché para revertir la transformación.")

    def has_stats(self, idx: int) -> bool:
        return False

    def clear_cache(self) -> None:
        pass