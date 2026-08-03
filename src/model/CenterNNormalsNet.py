import torch
from torch import nn

from src.model.decoders.center_prediction_head import CenterPredictionHead
from src.model.decoders.prediction_head import PredictionHead
from src.model.encoders.pointnet import PointNetEncoder


class CenterNNormalsNet(nn.Module):
    def __init__(
            self,
            amount_of_plane_normals_predicted=10,
            amount_of_axis_discrete_normals_predicted=10,
            amount_of_axis_continue_normals_predicted=10,
            use_bn=False,
            normalize_normals=False,
            encoder: str = "pointnet",
            n_points: int = 8192,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.n_points = n_points
        self.normalize_normals = normalize_normals
        self.amount_plane_normals = amount_of_plane_normals_predicted
        self.amount_axis_discrete_normals = amount_of_axis_discrete_normals_predicted
        self.amount_axis_continue_normals = amount_of_axis_continue_normals_predicted

        if encoder == "pointnet":
            self.encoder = PointNetEncoder(use_bn=self.use_bn)
            self.encoder_output_size = 1024
        else:
            raise ValueError("Encoder no soportado")

        # nx ny nz & confidence
        self.plane_normals_heads = nn.ModuleList(
            [PredictionHead(input_size=self.encoder_output_size, output_size=4, use_bn=self.use_bn) for _ in range(self.amount_plane_normals)]
        )

        # nx ny nz theta & confidence
        self.axis_discrete_normals_heads = nn.ModuleList(
            [PredictionHead(input_size=self.encoder_output_size, output_size=5, use_bn=self.use_bn) for _ in range(self.amount_axis_discrete_normals)]
        )

        # nx ny nz & confidence
        self.axis_continue_normals_heads = nn.ModuleList(
            [PredictionHead(input_size=self.encoder_output_size, output_size=4, use_bn=self.use_bn) for _ in range(self.amount_axis_continue_normals)]
        )

        self.center_prediction_head = CenterPredictionHead(input_size=self.encoder_output_size, use_bn=self.use_bn)

    def _forward_plane_normals(self, batch_size, pcd_features, center):
        plane_normals_list = []
        for head in self.plane_normals_heads:
            plane_normals_list.append(head(pcd_features))

        plane_normals = (torch.vstack(plane_normals_list).view(
            batch_size, self.amount_plane_normals, 4
        ))
        plane_predictions = torch.concat(
            (plane_normals, center.repeat(1, self.amount_plane_normals, 1)), dim=2
        )

        reorder_planes = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=plane_predictions.device).long()
        plane_predictions = plane_predictions[:, :, reorder_planes]
        plane_predictions[:, :, -1] = torch.sigmoid(plane_predictions[:, :, -1])

        if self.normalize_normals:
            plane_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                plane_predictions[:, :, 0:3].clone(), dim=2
            )

        return plane_predictions

    def _forward_axis_discrete_normals(self, batch_size, pcd_features, center):
        axis_discrete_normals_list = []

        for head in self.axis_discrete_normals_heads:
            axis_discrete_normals_list.append(head(pcd_features))

        axis_discrete_normals = (torch.vstack(axis_discrete_normals_list).view(
            batch_size, self.amount_axis_discrete_normals, 5
        ))
        axis_discrete_predictions = torch.concat(
            (axis_discrete_normals, center.repeat(1, self.amount_axis_discrete_normals, 1)), dim=2
        )
        # 0, 1, 2 normal vector
        # 3 is the angle
        # 4 is the confidence
        # 5 6 7 is the center
        reorder_axis_discrete = torch.tensor([0, 1, 2, 5, 6, 7, 3, 4], device=axis_discrete_predictions.device).long()
        axis_discrete_predictions = axis_discrete_predictions[:, :, reorder_axis_discrete]
        axis_discrete_predictions[:, :, -1] = torch.sigmoid(axis_discrete_predictions[:, :, -1])

        if self.normalize_normals:
            axis_discrete_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                axis_discrete_predictions[:, :, 0:3].clone(), dim=2
            )

        return axis_discrete_predictions

    def _forward_axis_continue(self, batch_size, pcd_features, center):
        axis_continue_normals_list = []

        for head in self.axis_continue_normals_heads:
            axis_continue_normals_list.append(head(pcd_features))

        axis_continue_normals = (torch.vstack(axis_continue_normals_list).view(
            batch_size, self.amount_axis_continue_normals, 4
        ))
        axis_continue_predictions = torch.concat(
            (axis_continue_normals, center.repeat(1, self.amount_axis_continue_normals, 1)), dim=2
        )

        reorder_planes = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=axis_continue_predictions.device).long()
        axis_continue_predictions = axis_continue_predictions[:, :, reorder_planes]
        axis_continue_predictions[:, :, -1] = torch.sigmoid(axis_continue_predictions[:, :, -1])

        if self.normalize_normals:
            axis_continue_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                axis_continue_predictions[:, :, 0:3].clone(), dim=2
            )

        return axis_continue_predictions

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)

        center = self.center_prediction_head(x).unsqueeze(dim=1)

        plane_predictions = (self._forward_plane_normals(batch_size, x, center)
                             if self.amount_plane_normals > 0 else None)

        axis_discrete_predictions = (self._forward_axis_discrete_normals(batch_size, x, center)
                                     if self.amount_axis_discrete_normals > 0 else None)

        axis_continue_predictions = (self._forward_axis_continue(batch_size, x, center)
                                     if self.amount_axis_continue_normals > 0 else None)

        return plane_predictions, axis_discrete_predictions, axis_continue_predictions
