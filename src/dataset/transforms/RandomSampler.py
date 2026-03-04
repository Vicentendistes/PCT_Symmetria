from typing import Optional
import torch

from src.dataset.transforms.AbstractTransform import AbstractTransform


class RandomSampler(AbstractTransform):
    def __init__(self, sample_size: int = 1024, keep_copy: bool = True):
        self.sample_size = sample_size
        self.keep_copy = keep_copy
        self.points_copy = None

    def transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.keep_copy:
            self.points_copy = points.clone()
        chosen_points = torch.randint(high=points.shape[0], size=(self.sample_size,))
        sample = points[chosen_points]
        return idx, sample, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    def inverse_transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.keep_copy:
            return idx, self.points_copy, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
        else:
            return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
