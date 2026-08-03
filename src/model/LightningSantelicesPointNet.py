import math

import lightning
import torch

from src.metrics.eval_script import (
    calculate_metrics_from_predictions,
    get_match_sequence_plane_symmetry,
)
from src.model.baselines.SantelicesPointNetGlobal import (
    SantelicesPointNetGlobal,
)
from src.model.losses.ConfidenceLoss import ConfidenceLoss
from src.model.losses.DistanceLoss import DistanceLoss
from src.model.losses.NormalLoss import NormalLoss
from src.model.losses.ReflectionSymmetryDistance import (
    ReflectionSymmetryDistance,
)
from src.model.losses.ReflectionSymmetryLoss import ReflectionSymmetryLoss
from src.model.matchers.SimpleMatcher import SimpleMatcher
from src.model.matchers.cost_matrix_methods import calculate_cost_matrix_normals


class LightningSantelicesPointNet(lightning.LightningModule):
    """HDF5-compatible, planar-only adaptation of the Santelices baseline."""

    LOSS_COMPONENT_NAMES = (
        "confidence",
        "normal",
        "center",
        "reflection_distance",
    )

    def __init__(
        self,
        amount_of_plane_normals_predicted: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        confidence_weight: float = 1.0,
        normal_weight: float = 1.0,
        center_weight: float = 1.0,
        reflection_symmetry_distance_weight: float = 0.1,
        use_bn: bool = False,
        normalize_normals: bool = True,
        epsilon_rate: float = 0.01,
        angle_threshold_degrees: float = 1.0,
        confidence_threshold: float = 0.01,
        compute_train_metrics: bool = False,
        compute_val_metrics: bool = False,
        compute_test_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.net = SantelicesPointNetGlobal(
            amount_of_plane_normals_predicted=(
                amount_of_plane_normals_predicted
            ),
            use_bn=use_bn,
            normalize_normals=normalize_normals,
        )
        self.loss_fn = ReflectionSymmetryLoss(
            confidence_weight=confidence_weight,
            confidence_loss=ConfidenceLoss(weighted=False),
            normal_weight=normal_weight,
            normal_loss=NormalLoss(check_normalized=True, reduction="mean"),
            distance_weight=center_weight,
            distance_loss=DistanceLoss(p=1, reduction="mean"),
            reflection_symmetry_distance_weight=(
                reflection_symmetry_distance_weight
            ),
            reflection_symmetry_distance=ReflectionSymmetryDistance(
                p=1,
                reduction="mean",
            ),
        )
        self.matcher = SimpleMatcher(
            calculate_cost_matrix_normals,
            device="cpu",
        )

        # SymPlane compares 1 - |dot(n1, n2)|, not an angle in radians.
        self.metric_params = {
            "eps": epsilon_rate,
            "theta": 1.0 - math.cos(math.radians(angle_threshold_degrees)),
            "confidence_threshold": confidence_threshold,
            "rot_angle_threshold": math.radians(1.0),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(
                f"Expected a rank-3 point tensor, received {points.ndim}"
            )
        if points.shape[1] != 3:
            points = points.transpose(1, 2)
        return self.net(points.float())

    def _should_compute_metrics(self, stage: str) -> bool:
        return bool(getattr(self.hparams, f"compute_{stage}_metrics"))

    def _step(self, batch, stage: str) -> torch.Tensor:
        batch.device = self.device
        self.matcher.device = self.device

        points = torch.stack(batch.get_points()).to(self.device).float()
        predictions = self.net(points.transpose(1, 2))
        ground_truth = batch.get_plane_syms()

        c_hat, matched_pred, matched_true, _, _ = (
            self.matcher.get_optimal_assignment(
                batch.get_points(),
                predictions,
                ground_truth,
            )
        )
        loss, components = self.loss_fn(
            (batch, predictions, c_hat, matched_pred, matched_true)
        )

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch.size,
            sync_dist=True,
        )
        for name, value in zip(
            self.LOSS_COMPONENT_NAMES,
            components / batch.size,
        ):
            self.log(
                f"{stage}_loss_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch.size,
                sync_dist=True,
            )

        if self._should_compute_metrics(stage):
            mean_ap, phc, _ = calculate_metrics_from_predictions(
                [(batch.get_points(), predictions, ground_truth)],
                get_match_sequence_plane_symmetry,
                self.metric_params,
            )
            self.log(
                f"{stage}_mAP",
                mean_ap,
                on_step=False,
                on_epoch=True,
                batch_size=batch.size,
                sync_dist=True,
            )
            self.log(
                f"{stage}_PHC",
                phc,
                on_step=False,
                on_epoch=True,
                batch_size=batch.size,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch.device = self.device
        points = torch.stack(batch.get_points()).to(self.device).float()
        predictions = self.net(points.transpose(1, 2))
        return batch, predictions, None, None
