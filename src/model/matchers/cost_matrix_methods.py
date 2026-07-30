import torch

from src.utils.plane import SymPlane


def calculate_cost_matrix_normals(points, y_pred, y_true):
    """
    :param points: N x 3
    :param y_pred: K x 7
    :param y_true: M x 6
    :return: K x M
    """
    normals_pred = y_pred[:, 0:3]
    normals_true = y_true[:, 0:3]

    return 1 - torch.abs(normals_pred @ normals_true.T)


def calculate_cost_matrix_sde(points, y_pred, y_true):
    """

    :param points: N x 3
    :param y_pred: K x 7
    :param y_true: M x 6
    :return: K x M
    """
    cost_matrix = torch.zeros(y_pred.shape[0], y_true.shape[0])

    for i in range(y_pred.shape[0]):
        for j in range(y_true.shape[0]):
            pred_plane = SymPlane.from_tensor(y_pred[i, 0:6])
            true_plane = SymPlane.from_tensor(y_true[j, 0:6])
            cost_matrix[i, j] = torch.abs(
                torch.norm(
                    true_plane.reflect_points(points) - pred_plane.reflect_points(points),
                    dim=0
                )
            ).sum()
            if cost_matrix[i, j].isnan().any():
                print("Got cost matrix nan with planes:")
                print("pred", pred_plane)
                print("true", true_plane)

    return cost_matrix