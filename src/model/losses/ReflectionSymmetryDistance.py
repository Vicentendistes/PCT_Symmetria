import torch
from torch import nn
from torch import linalg as alg

from src.utils.plane import SymPlane

REDUCTIONS = {
    "mean": torch.mean,
    "sum": torch.sum,
}


# Using old code
def get_sde(points, pred_plane, true_plane, p=2):
    """
    :param points:
    :param pred_plane:
    :param true_plane:
    :param p:
    :return:
    """
    pred_plane = SymPlane.from_tensor(pred_plane)
    true_plane = SymPlane.from_tensor(true_plane, normalize=True)

    # Notice:
    #   Here using dim=0 is equal to optimizing the mean
    #   difference between both transformed point clouds component per component.
    #   I'm accumulating the reconstruction error per x, y and z.
    #   Other option was accumulate the reconstruction error per point.
    #       dim=0 returns a vector of shape 3 where each component is the size of the error per axis.
    #       dim=1 returns a vector of shape N where each component is the size of the error per point.
    #   Weirdly enough only dim=0 converges on training.
    # It a different version of a haudsorf distance within all points instead only the minimum

    # Pointnet -> conjunto critico -> 1024 o menos -> chamfer converger mejor
    # (A, B, 0) / 3 => (A, B) -> optimizar en este caso

    return torch.norm(
        true_plane.reflect_points(points) - pred_plane.reflect_points(points),
        dim=0, p=p
    ).mean()


class ReflectionSymmetryDistance(nn.Module):
    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self,
                points,
                normal_pred, normal_true,
                center_pred, center_true
                ):
        """

        :param points:
        :param y_pred: M x 6
        :param y_true: M x 6
        :return:
        """
        y_pred = torch.cat((normal_pred, center_pred), dim=1)
        y_true = torch.cat((normal_true, center_true), dim=1)

        m = y_pred.shape[0]
        loss = torch.tensor([0.0], device=y_pred.device)
        for i in range(m):
            loss += get_sde(points, y_pred[i], y_true[i])
        return loss / m


class ReflectionSymmetryDistanceNewUNUSED(nn.Module):
    def __init__(self, p=1, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self,
                points,
                normal_pred, normal_true,
                center_pred, center_true
                ):
        m = normal_true.shape[0]
        n = points.shape[0]

        points = points.repeat(m, 1)

        normal_true = normal_true.repeat(n, 1)
        normal_pred = normal_pred.repeat(n, 1)

        center_true = center_true.repeat(n, 1)
        center_pred = center_pred.repeat(n, 1)

        offset_true = - alg.vecdot(normal_true, center_true)
        offset_pred = - alg.vecdot(normal_pred, center_pred)

        distances_true = alg.vecdot(points, normal_true) + offset_true
        distances_pred = alg.vecdot(points, normal_pred) + offset_pred

        distances_true = distances_true.unsqueeze(-1).repeat(1, 3)
        distances_pred = distances_pred.unsqueeze(-1).repeat(1, 3)

        reflected_true = points - 2 * distances_true * normal_true
        reflected_pred = points - 2 * distances_pred * normal_pred

        distance_between_reflected = torch.norm(reflected_pred - reflected_true, p=self.p, dim=1)

        return REDUCTIONS[self.reduction](distance_between_reflected)


if __name__ == "__main__":
    m = 6
    n = 14_440
    points = torch.rand((n, 3))
    y_pred = torch.rand((m, 7))
    y_true = torch.rand((m, 6))

    normal_pred = torch.nn.functional.normalize(y_pred[:, 0:3], dim=1)
    normal_true = torch.nn.functional.normalize(y_true[:, 0:3], dim=1)

    center_pred = y_pred[:, 3:6]
    center_true = y_true[:, 3:6]

    old = ReflectionSymmetryDistance()
    new = ReflectionSymmetryDistanceNewUNUSED(p=2)

    old_result = old.forward(points, normal_pred, normal_true, center_pred, center_true)

    new_result = new.forward(points, normal_pred, normal_true, center_pred, center_true)

    # they are almost the same
    print("OLD")
    print(old_result)
    print("NEW")
    print(new_result)
