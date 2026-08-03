from typing import Callable, Union

import lightning
import torch

from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry, \
    get_match_sequence_continue_rotational_symmetry, \
    get_match_sequence_discrete_rotational_symmetry
from src.model.CenterNNormalsNet import CenterNNormalsNet
from src.model.losses.ConfidenceLoss import ConfidenceLoss
from src.model.losses.DistanceLoss import DistanceLoss
from src.model.losses.NormalLoss import NormalLoss
from src.model.losses.ReflectionSymmetryDistance import ReflectionSymmetryDistance
from src.model.losses.ReflectionSymmetryLoss import ReflectionSymmetryLoss
from src.model.matchers.SimpleMatcher import SimpleMatcher
from src.model.matchers.cost_matrix_methods import calculate_cost_matrix_normals


class LightingCenterNNormalsNet(lightning.LightningModule):
    def __init__(self,
                 amount_of_plane_normals_predicted: int = 32,
                 amount_of_axis_discrete_normals_predicted: int = 16,
                 amount_of_axis_continue_normals_predicted: int = 16,
                 plane_loss: Union[ReflectionSymmetryLoss, str] = "default",
                 w1: float = 1.0,
                 w2: float = 1.0,
                 w3: float = 1.0,
                 eps: float = 0.01,
                 theta: float = 0.00015230484,  # 1° between axis/normals
                 confidence_threshold: float = 0.01,
                 rot_angle_threshold: float = 0.0174533,  # 1° of difference between rot angles
                 cost_matrix_method: Callable = calculate_cost_matrix_normals,
                 print_losses: bool = False,
                 use_bn: bool = False,
                 normalize_normals: bool = True,
                 encoder: str = "pointnet",
                 n_points: int = 8192
                 ):
        super().__init__()
        self.use_bn = use_bn
        self.n_points = n_points
        self.encoder_used = encoder
        self.normalize_normals = normalize_normals
        self.print_losses = print_losses
        self.cost_matrix_method = cost_matrix_method
        self.matcher = SimpleMatcher(self.cost_matrix_method, self.device)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        if plane_loss == "default":
            self.plane_loss = ReflectionSymmetryLoss(
                confidence_weight=1.0, confidence_loss=ConfidenceLoss(),
                normal_weight=1.0, normal_loss=NormalLoss(),
                distance_weight=1.0, distance_loss=DistanceLoss(),
                reflection_symmetry_distance_weight=0.1,
                reflection_symmetry_distance=ReflectionSymmetryDistance()
            )
        else:
            self.plane_loss = plane_loss
        self.plane_loss_tag = [
            "confidence",
            "normal",
            "distance",
            "ref_sym_distance",
        ]


        self.net = CenterNNormalsNet(
            amount_of_plane_normals_predicted,
            amount_of_axis_discrete_normals_predicted,
            amount_of_axis_continue_normals_predicted,
            use_bn=self.use_bn,
            normalize_normals=self.normalize_normals,
            encoder=encoder,
            n_points=self.n_points
        )
        self.eps = eps
        self.theta = theta
        self.confidence_threshold = confidence_threshold
        self.rot_angle_threshold = rot_angle_threshold
        self.metric_param_dict = {
            "eps": self.eps,
            "theta": self.theta,
            "confidence_threshold": self.confidence_threshold,
            "rot_angle_threshold": self.rot_angle_threshold,
        }

        # If warning concerns you read this https://github.com/Lightning-AI/pytorch-lightning/discussions/13615
        # Honestly idk will leave it like this for now
        self.save_hyperparameters(ignore=["net"]) # , "plane_loss", "discrete_rotational_loss", "continue_rotational_loss"

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _log(
            self, metric_val, metric_name, sym_tag, step_tag, batch_size,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
    ):
        self.log(f"{sym_tag}_{step_tag}_{metric_name}", metric_val, on_step=on_step, on_epoch=on_epoch,
                 prog_bar=prog_bar, batch_size=batch_size, sync_dist=sync_dist)

    def _process_prediction(self,
                            batch, sym_pred, sym_true,
                            loss_fun, sym_tag, step_tag,
                            losses_tags, metrics_match_sequence_fun):
        c_hat, match_pred, match_true, pred2true, true2pred = self.matcher.get_optimal_assignment(batch.get_points(),
                                                                                                  sym_pred, sym_true)
        bundled_predictions = (batch, sym_pred, c_hat, match_pred, match_true)
        loss, others = loss_fun(bundled_predictions)

        eval_predictions = [(batch.get_points(), sym_pred, sym_true)]
        map, phc, pr_curve = calculate_metrics_from_predictions(eval_predictions, metrics_match_sequence_fun,
                                                                self.metric_param_dict)

        for idx in range(others.shape[0]):
            self._log(others[idx], f"loss_{losses_tags[idx]}", sym_tag, step_tag, batch.size)

        self._log(loss, "loss", sym_tag, step_tag, batch.size)
        self._log(map, "map", sym_tag, step_tag, batch.size)
        self._log(phc, "phc", sym_tag, step_tag, batch.size)

        return loss, map, phc

    def _step(self, batch, step_tag):
        batch.device = self.device
        self.matcher.device = self.device
        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)
        loss = torch.tensor(0.0, device=points.device)

        if plane_predictions is not None:
            plane_loss, plane_map, plane_phc = self._process_prediction(
                batch, plane_predictions, batch.get_plane_syms(), self.plane_loss,
                "plane", step_tag, self.plane_loss_tag, get_match_sequence_plane_symmetry,
            )
            loss += plane_loss * self.w1

        if axis_discrete_predictions is not None:
            discrete_axis_loss, map_discrete_axis, phc_discrete_axis = self._process_prediction(
                batch, axis_discrete_predictions, batch.get_axis_discrete_syms(), self.discrete_rotational_loss,
                "d_axis", step_tag, self.discrete_rotational_loss_tag, get_match_sequence_discrete_rotational_symmetry
            )
            loss += discrete_axis_loss * self.w2

        if axis_continue_predictions is not None:
            continue_axis_loss, map_continue_axis, phc_continue_axis = self._process_prediction(
                batch, axis_continue_predictions, batch.get_axis_continue_syms(), self.continue_rotational_loss,
                "c_axis", step_tag, self.continue_rotational_loss_tag, get_match_sequence_continue_rotational_symmetry

            )
            loss += continue_axis_loss * self.w3

        self._log(loss, "loss", "total", step_tag, batch.size, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch.device = self.device
        self.matcher.device = self.device

        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)

        return batch, plane_predictions, axis_discrete_predictions, axis_continue_predictions

    def on_after_backward(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                if param.grad.isnan().any():
                    print(f"{name} got nan!")
