# src/dataset/transforms/UnitSphereNormalization.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch

from src.dataset.transforms.AbstractTransform import AbstractTransform


class UnitSphereNormalization(AbstractTransform):
    """
    Centra los puntos en su centroide y los escala para que el
    radio máximo sea 1 (esfera unitaria). Guarda (centroid, scale)
    *por índice* para poder revertir más tarde de forma segura.

    Convención de simetrías: columnas [3:6] contienen un punto
    (p.ej. centro en el plano/eje) y por eso se trasladan/escalan.
    Las normales (nx,ny,nz) NO se tocan aquí.
    """

    def __init__(self, eps: float = 1e-9, keep_cache: bool = False) -> None:
        """
        :param eps: evita división por cero si la nube colapsa.
        :param keep_cache: si True, conserva el cache tras inverse_transform;
                           si False (default), lo elimina al revertir.
        """
        self.eps = eps
        self.keep_cache = keep_cache
        # cache: idx -> (centroid [3], farthest_distance [1])
        self._stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ---------- helpers internos ----------

    def _compute_stats(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centroid = points.mean(dim=0)                            # [3]
        shifted = points - centroid
        farthest = torch.max(torch.linalg.norm(shifted, dim=1))  # escalar
        # clamp para evitar división por cero
        scale = torch.clamp(farthest, min=self.eps)
        return centroid, scale

    def _normalize_points(self, points: torch.Tensor, centroid: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (points - centroid) / scale

    def _denormalize_points(self, points: torch.Tensor, centroid: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return points * scale + centroid

    def _normalize_syms(self, syms: torch.Tensor, centroid: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # mueve sólo las columnas 3:6 (punto asociado)
        syms = syms.clone()
        syms[:, 3:6] = (syms[:, 3:6] - centroid) / scale
        return syms

    def _denormalize_syms(self, syms: torch.Tensor, centroid: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        syms = syms.clone()
        syms[:, 3:6] = syms[:, 3:6] * scale + centroid
        return syms

    # ---------- API AbstractTransform ----------

    def transform(
        self,
        idx: int,
        points: torch.Tensor,
        planar_symmetries: Optional[torch.Tensor],
        axis_continue_symmetries: Optional[torch.Tensor],
        axis_discrete_symmetries: Optional[torch.Tensor],
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # calcula y guarda stats por ítem
        centroid, scale = self._compute_stats(points)
        self._stats[idx] = (centroid.detach(), scale.detach())

        # normaliza puntos y simetrías (si existen)
        points = self._normalize_points(points, centroid, scale)
        if planar_symmetries is not None:
            planar_symmetries = self._normalize_syms(planar_symmetries, centroid, scale)
        if axis_continue_symmetries is not None:
            axis_continue_symmetries = self._normalize_syms(axis_continue_symmetries, centroid, scale)
        if axis_discrete_symmetries is not None:
            axis_discrete_symmetries = self._normalize_syms(axis_discrete_symmetries, centroid, scale)

        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    def inverse_transform(
        self,
        idx: int,
        points: torch.Tensor,
        planar_symmetries: Optional[torch.Tensor],
        axis_continue_symmetries: Optional[torch.Tensor],
        axis_discrete_symmetries: Optional[torch.Tensor],
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if idx not in self._stats:
            raise RuntimeError(f"No hay stats guardadas para idx={idx}. ¿Llamaste transform() antes?")

        centroid, scale = self._stats[idx]
        # asegura que centroid/scale estén en el mismo device
        centroid = centroid.to(points.device)
        scale = scale.to(points.device)

        points = self._denormalize_points(points, centroid, scale)
        if planar_symmetries is not None:
            planar_symmetries = self._denormalize_syms(planar_symmetries, centroid, scale)
        if axis_continue_symmetries is not None:
            axis_continue_symmetries = self._denormalize_syms(axis_continue_symmetries, centroid, scale)
        if axis_discrete_symmetries is not None:
            axis_discrete_symmetries = self._denormalize_syms(axis_discrete_symmetries, centroid, scale)

        if not self.keep_cache:
            self._stats.pop(idx, None)  # libera memoria por ítem

        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    # utilidades opcionales
    def has_stats(self, idx: int) -> bool:
        return idx in self._stats

    def clear_cache(self) -> None:
        self._stats.clear()
