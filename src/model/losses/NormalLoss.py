import warnings

import torch
from torch import nn

REDUCTIONS = {
    "mean": torch.mean,
    "sum": torch.sum,
}

def calculate_angle_loss(y_pred, y_true):
    """
    :param y_pred: M x 6
    :param y_true: M x 6
    :return:
    """
    normals_pred = torch.nn.functional.normalize(y_pred[:, 0:3], dim=1)  # M x 3
    normals_true = torch.nn.functional.normalize(y_true[:, 0:3], dim=1)  # M x 3

    # cos(theta) = n_1 . n_2.T
    # => if n_1 == n_2 => cos(theta) = 1
    # or if n_1 == -n_2 => cos(Theta) = -1
    # Min theta <=> Min 1 - |cos(Theta)| <=> 1 - |n_1 . n_2.T|
    cos_angle = 1 - torch.abs(normals_true @ normals_pred.T)
    return cos_angle.min(dim=0).values.mean()


class NormalLoss(nn.Module):
    def __init__(self, check_normalized=True, reduction="mean"):
        super().__init__()
        self.check_normalized = check_normalized
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        normals_pred = torch.nn.functional.normalize(y_pred[:, 0:3], dim=1)  # M x 3
        normals_true = torch.nn.functional.normalize(y_true[:, 0:3], dim=1)  # M x 3

        # cos(theta) = n_1 . n_2.T
        # => if n_1 == n_2 => cos(theta) = 1
        # or if n_1 == -n_2 => cos(Theta) = -1
        # Min theta <=> Min 1 - |cos(Theta)| <=> 1 - |n_1 . n_2.T|
        cos_angle = 1 - torch.abs(normals_true @ normals_pred.T)
        return cos_angle.min(dim=0).values.mean()

    def forward_2(self, n_pred, n_true):
        if self.check_normalized:
            pred_normalized = torch.allclose(torch.norm(n_pred, dim=1),
                                             torch.ones((n_pred.shape[0]), device=n_pred.device))
            true_normalized = torch.allclose(torch.norm(n_true, dim=1),
                                             torch.ones((n_true.shape[0]), device=n_true.device))
            if not pred_normalized:
                warnings.warn("Got n_pred with normals that are not normalized!")
            if not true_normalized:
                warnings.warn("Got n_true with normals that are not normalized!")

        # 1 - |a.b/(|a||b|))| < 0.00015230484 is good
        # |a.b/(|a||b|) = cos(angle between a and b)
        abs_angle = torch.abs(
            torch.linalg.vecdot(n_true, n_pred) / (torch.norm(n_true, dim=1) * torch.norm(n_pred, dim=1))
        )

        return REDUCTIONS[self.reduction](1 - abs_angle)
