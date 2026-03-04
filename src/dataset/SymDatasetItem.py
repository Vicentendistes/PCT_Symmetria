from copy import deepcopy
from typing import Optional

import torch

from src.dataset.transforms.AbstractTransform import AbstractTransform
from src.dataset.transforms.ReverseTransform import ReverseTransform

SHAPE_TYPE = {
    "astroid": 0,
    "citrus": 1,
    "cylinder": 2,
    "egg_keplero": 3,
    "geometric_petal": 4,
    "lemniscate": 5,
    "m_convexities": 6,
    "mouth_curve": 7,
    "revolution": 8,
    "square": 9,
}
SHAPE_TYPE_AMOUNT = len(SHAPE_TYPE.keys())

PERTURBATION_TYPE = {
    'clean': 0,
    'uniform': 1,
    'gaussian': 2,
    'undersampling': 3,
    'undersampling+uniform': 4,
    'undersampling+gaussian': 5,
}
PERTURBATION_TYPE_AMOUNT = len(PERTURBATION_TYPE.keys())


class SymDatasetItem:
    def __init__(
            self,
            filename: str,
            idx: int,
            points: torch.tensor,
            plane_symmetries: Optional[torch.tensor],
            axis_continue_symmetries: Optional[torch.tensor],
            axis_discrete_symmetries: Optional[torch.tensor],
            transform: AbstractTransform,
    ):
        self.filename = filename
        self.shape_type = filename.split("-")[1]
        self.perturbation_type = filename.split("-")[2]
        self.transform = transform

        self.idx = idx
        self.points = points
        self.plane_symmetries = plane_symmetries
        self.axis_continue_symmetries = axis_continue_symmetries
        self.axis_discrete_symmetries = axis_discrete_symmetries

    def get_item_elements(self):
        return (
            self.idx,
            self.points,
            self.plane_symmetries,
            self.axis_continue_symmetries,
            self.axis_discrete_symmetries,
        )

    def get_untransformed_item(self):
        reverse_transform = ReverseTransform(deepcopy(self.transform))
        item_elements = reverse_transform(
            *self.get_item_elements()
        )

        return SymDatasetItem(
            self.filename,
            *item_elements,
            reverse_transform
        )

    def get_shape_type_classification_label(self, device="cpu"):
        label = torch.zeros(SHAPE_TYPE_AMOUNT, device=device)
        label[SHAPE_TYPE[self.shape_type]] = 1
        return label

    def __repr__(self):
        str_rep = ""
        str_rep += f"SymmetryDatasetItem N°{self.idx}\n"
        str_rep += f"\tFilename: {self.filename}\n"
        str_rep += f"\tElements Shape:\n"
        str_rep += f"\t\tPoints: {self.points.shape}\n"
        if self.plane_symmetries is None:
            str_rep += f"\t\tPlane Syms: Not present.\n"
        else:
            str_rep += f"\t\tPlane Syms: {self.plane_symmetries.shape}\n"
        if self.axis_discrete_symmetries is None:
            str_rep += f"\t\tAxis Discrete Syms: Not present.\n"
        else:
            str_rep += f"\t\tAxis Discrete Syms: {self.axis_discrete_symmetries.shape}\n"
        if self.axis_continue_symmetries is None:
            str_rep += f"\t\tAxis Continue Syms: Not present.\n"
        else:
            str_rep += f"\t\tAxis Continue Syms: {self.axis_continue_symmetries.shape}\n"
        return str_rep

    def to(self, device):
        self.points = self.points.to(device)
        self.plane_symmetries = self.plane_symmetries.to(device) if self.plane_symmetries is not None else None
        self.axis_continue_symmetries = self.axis_continue_symmetries.to(
            device) if self.axis_continue_symmetries is not None else None
        self.axis_discrete_symmetries = self.axis_discrete_symmetries.to(
            device) if self.axis_discrete_symmetries is not None else None
        return self

    def visualize(self):
        # TODO: Implement this.
        pass
