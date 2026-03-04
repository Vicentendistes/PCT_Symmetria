# src/dataset/transforms/FarthestPointSampling.py
from typing import Optional, Tuple
import torch
from src.dataset.transforms.AbstractTransform import AbstractTransform

class FarthestPointSampling(AbstractTransform):
    """
    Selecciona n_points puntos por Farthest Point Sampling (FPS).
    No modifica las simetrías (solo submuestrea los puntos).
    """
    def __init__(self, n_points: int, start: str = "centroid"):
        """
        :param n_points: cantidad de puntos objetivo tras el muestreo
        :param start: 'centroid' (comienza con el más alejado del centroide) o 'random'
        """
        self.n_points = n_points
        self.start = start

    @torch.no_grad()
    def _fps_indices(self, pts: torch.Tensor, n: int) -> torch.Tensor:
        """
        pts: [N,3] en el *mismo* device que usarás para computar.
        return: idxs [n] (long)
        """
        N = pts.shape[0]
        n = min(n, N)
        idxs = torch.empty(n, dtype=torch.long, device=pts.device)

        if self.start == "random":
            farthest = torch.randint(0, N, (1,), device=pts.device).item()
        else:
            c = pts.mean(dim=0, keepdim=True)           # [1,3]
            d = torch.norm(pts - c, dim=1)              # [N]
            farthest = torch.argmax(d).item()

        # distancia mínima a conjunto de seleccionados
        min_dist = torch.full((N,), float("inf"), device=pts.device)

        for i in range(n):
            idxs[i] = farthest
            diff = pts - pts[farthest]                  # [N,3]
            dist = (diff * diff).sum(dim=1)             # distancia^2 a último seleccionado
            min_dist = torch.minimum(min_dist, dist)    # mantener la mínima a algún seleccionado
            farthest = torch.argmax(min_dist).item()

        return idxs

    def transform(
        self,
        idx: int,
        points: torch.Tensor,
        planar_symmetries: Optional[torch.Tensor],
        axis_continue_symmetries: Optional[torch.Tensor],
        axis_discrete_symmetries: Optional[torch.Tensor],
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # points: [N,3]
        device = points.device
        sel = self._fps_indices(points, self.n_points)     # [n_points]
        points = points.index_select(0, sel).to(device)     # submuestreo
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    # FPS no necesita “inverse_transform” (no podemos recuperar los puntos descartados)
    def inverse_transform(
        self,
        idx: int,
        points: torch.Tensor,
        planar_symmetries: Optional[torch.Tensor],
        axis_continue_symmetries: Optional[torch.Tensor],
        axis_discrete_symmetries: Optional[torch.Tensor],
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
