from typing import Optional
import torch

from src.dataset.transforms.AbstractTransform import AbstractTransform


class IdentityTransform(AbstractTransform):
    def inverse_transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    def transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
