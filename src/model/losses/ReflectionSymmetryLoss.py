import lzma

import numpy as np
import torch
from torch import nn

from src.dataset.SymDatasetBatcher import SymDatasetBatcher
from src.dataset.SymDatasetItem import SymDatasetItem
from src.dataset.transforms.IdentityTransform import IdentityTransform
from src.model.losses.ConfidenceLoss import ConfidenceLoss
from src.model.losses.DistanceLoss import DistanceLoss
from src.model.losses.NormalLoss import NormalLoss
from src.model.losses.ReflectionSymmetryDistance import ReflectionSymmetryDistance
from src.model.matchers.SimpleMatcher import SimpleMatcher
from src.model.matchers.cost_matrix_methods import calculate_cost_matrix_normals


class ReflectionSymmetryLoss(nn.Module):
    def __init__(
            self,
            confidence_weight: float, confidence_loss: ConfidenceLoss,
            normal_weight: float, normal_loss: NormalLoss,
            distance_weight: float, distance_loss: DistanceLoss,
            reflection_symmetry_distance_weight: float, reflection_symmetry_distance: ReflectionSymmetryDistance
    ):
        super().__init__()

        self.confidence_weight = confidence_weight
        self.normal_weight = normal_weight
        self.distance_weight = distance_weight
        self.reflection_symmetry_distance_weight = reflection_symmetry_distance_weight

        self.confidence_loss = confidence_loss
        self.normal_loss = normal_loss
        self.distance_loss = distance_loss
        self.reflection_symmetry_distance = reflection_symmetry_distance

    def forward(self, bundled_plane_predictions):
        batch, plane_predictions, plane_c_hats, matched_plane_pred, matched_plane_real = bundled_plane_predictions

        batch_size = batch.size
        loss_matrix = torch.zeros((batch_size, 4), device=batch.device)

        for b_idx in range(batch_size):
            item = batch.get_item(b_idx)

            curr_points = item.points

            curr_y_true = matched_plane_real[b_idx]
            curr_y_pred = matched_plane_pred[b_idx]

            curr_conf_true = plane_c_hats[b_idx]
            curr_conf_pred = plane_predictions[b_idx, :, -1]
            conf_loss = self.confidence_weight * self.confidence_loss(curr_conf_pred, curr_conf_true)

            if curr_y_true is None:
                loss_matrix[b_idx, 0] = conf_loss
                continue

            curr_normal_true = curr_y_true[:, 0:3]
            curr_center_true = curr_y_true[:, 3:6]

            curr_normal_pred = curr_y_pred[:, 0:3]
            curr_center_pred = curr_y_pred[:, 3:6]

            normal_loss = self.normal_weight * self.normal_loss(curr_normal_pred, curr_normal_true)

            distance_loss = self.distance_weight * self.distance_loss(curr_center_pred, curr_center_true)

            reflection_symmetry_distance = (self.reflection_symmetry_distance_weight *
                                            self.reflection_symmetry_distance(
                                                curr_points,
                                                curr_normal_pred, curr_normal_true,
                                                curr_center_pred, curr_center_true
                                            )
                                            )

            loss_matrix[b_idx, 0] = conf_loss
            loss_matrix[b_idx, 1] = normal_loss
            loss_matrix[b_idx, 2] = distance_loss
            loss_matrix[b_idx, 3] = reflection_symmetry_distance

            total_loss = loss_matrix[b_idx].sum()

        loss = torch.sum(loss_matrix) / batch_size
        return loss, loss_matrix.sum(dim=0)


def my_parse_sym_file(filename, debug):
    planar_symmetries = []
    axis_continue_symmetries = []
    axis_discrete_symmetries = []

    with open(filename) as f:
        line_amount = int(f.readline())
        for _ in range(line_amount):
            line = f.readline().split(" ")
            line = [x.replace("\n", "") for x in line]
            if line[0] == "plane":
                plane = [float(x) for x in line[1::]]
                planar_symmetries.append(torch.tensor(plane))
            elif line[0] == "axis" and line[-1] == "inf":
                plane = [float(x) for x in line[1:7]]
                axis_continue_symmetries.append(torch.tensor(plane))
            else:
                plane = [float(x) for x in line[1::]]
                axis_discrete_symmetries.append(torch.tensor(plane))

    planar_symmetries = None if len(planar_symmetries) == 0 else torch.stack(planar_symmetries).float()
    axis_continue_symmetries = None if len(axis_continue_symmetries) == 0 else torch.stack(
        axis_continue_symmetries).float()
    axis_discrete_symmetries = None if len(axis_discrete_symmetries) == 0 else torch.stack(
        axis_discrete_symmetries).float()
    if debug:
        print(f'Parsed file at: {filename}')
        formatted_print = lambda probably_tensor, text: print(f'\t No {text} found.') if probably_tensor is None \
            else print(f'\tGot {probably_tensor.shape[0]} {text}.')
        formatted_print(planar_symmetries, "Plane symmetries")
        formatted_print(axis_discrete_symmetries, "Discrete axis symmetries")
        formatted_print(axis_continue_symmetries, "Continue axis symmetries")

    return planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries


def my_read_points(filename, debug) -> torch.Tensor:
    with lzma.open(filename, 'rb') as fhandle:
        points = torch.tensor(np.loadtxt(fhandle))

    if debug:
        torch.set_printoptions(linewidth=200)
        torch.set_printoptions(precision=3)
        torch.set_printoptions(sci_mode=False)
        print(f'[{filename}]: {points.shape = }\n{points = }')

    return points


if __name__ == "__main__":
    DEBUG = True
    FILENAME = "/data/sym-10k-xz-split-class-noparallel/train/citrus/000014-citrus-uniform"
    HEADS = 10
    BATCH_SIZE = 1

    y_true, _, _ = my_parse_sym_file(FILENAME + "-sym.txt", DEBUG)
    y_true = y_true.float()
    points = my_read_points(FILENAME + ".xz", DEBUG).float()

    y_pred = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ]).float()
    y_pred = y_pred.reshape(BATCH_SIZE, y_pred.shape[0], y_pred.shape[1])

    loss_obj = ReflectionSymmetryLoss(
        confidence_weight=1.0, confidence_loss=ConfidenceLoss(weighted=False),
        normal_weight=1.0, normal_loss=NormalLoss(),
        reflection_symmetry_distance_weight=0.1, reflection_symmetry_distance=ReflectionSymmetryDistance(),
        distance_weight=1.0, distance_loss=DistanceLoss()
    )

    data_item = SymDatasetItem(
        filename=FILENAME,
        idx=1,
        points=points,
        plane_symmetries=y_true,
        axis_continue_symmetries=None,
        axis_discrete_symmetries=None,
        transform=IdentityTransform()
    )

    batch = SymDatasetBatcher(item_list=[data_item])

    matcher = SimpleMatcher(method=calculate_cost_matrix_normals, device="cpu")

    c_hat, match_pred, match_true, pred2true, true2pred = matcher.get_optimal_assignment(
        batch.get_points(), y_pred, batch.get_plane_syms()
    )

    bundled_plane_predictions = batch, y_pred, c_hat, match_pred, match_true

    loss, loss_2 = loss_obj.forward(bundled_plane_predictions)

    print("conf", loss_2[0])
    print("sde", loss_2[3])
    print("angle", loss_2[1])
    print("distance", loss_2[2])
    print(loss)
