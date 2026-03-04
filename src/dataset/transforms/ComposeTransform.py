from typing import List, Optional
import torch

from src.dataset.transforms.AbstractTransform import AbstractTransform


class ComposeTransform(AbstractTransform):
    def __init__(
            self,
            transforms: List[AbstractTransform]
    ):
        self.transforms = transforms

    def inverse_transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        for a_transform in reversed(self.transforms):
            (idx, points, planar_symmetries,
             axis_continue_symmetries, axis_discrete_symmetries) = a_transform.inverse_transform(
                idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
            )
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    def transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        for a_transform in self.transforms:
            idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries = a_transform.transform(
                idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
            )
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
