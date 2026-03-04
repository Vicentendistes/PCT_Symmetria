from abc import ABC, abstractmethod
from typing import Optional
import torch


class AbstractTransform(ABC):

    @abstractmethod
    def transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass

    @abstractmethod
    def inverse_transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass

    def __call__(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.transform(idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries)
