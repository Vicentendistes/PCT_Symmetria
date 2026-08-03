import torch
import torch.nn.functional as F
from torch import nn

from src.model.decoders.center_prediction_head import CenterPredictionHead
from src.model.decoders.prediction_head import PredictionHead
from src.model.encoders.pointnet import PointNetEncoder


class SantelicesPointNetGlobal(nn.Module):
    """Planar-only version of the global PointNet model used by Santelices.

    The downloaded implementation uses ``vstack(...).view(B, M, 4)`` to
    assemble the independent prediction heads. That ordering is correct only
    when B == 1. This implementation uses ``stack(..., dim=1)`` and therefore
    preserves the same result for B == 1 while also supporting larger batches.
    """

    def __init__(
        self,
        amount_of_plane_normals_predicted: int = 32,
        use_bn: bool = False,
        normalize_normals: bool = True,
    ) -> None:
        super().__init__()

        if amount_of_plane_normals_predicted <= 0:
            raise ValueError("amount_of_plane_normals_predicted must be positive")

        self.amount_of_plane_normals_predicted = amount_of_plane_normals_predicted
        self.normalize_normals = normalize_normals

        self.encoder = PointNetEncoder(use_bn=use_bn)
        self.plane_heads = nn.ModuleList(
            [
                PredictionHead(input_size=1024, output_size=4, use_bn=use_bn)
                for _ in range(amount_of_plane_normals_predicted)
            ]
        )
        self.center_head = CenterPredictionHead(input_size=1024, use_bn=use_bn)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Return M global plane candidates as ``(B, M, 7)``.

        Each candidate is ``[nx, ny, nz, px, py, pz, confidence]``. All heads
        share the center predicted for the complete point cloud, matching the
        downloaded CenterNNormalsNet.
        """
        if points.ndim != 3 or points.shape[1] != 3:
            raise ValueError(
                "Expected points with shape (B, 3, N), "
                f"received {tuple(points.shape)}"
            )

        features = self.encoder(points)
        center = self.center_head(features)

        head_outputs = torch.stack(
            [head(features) for head in self.plane_heads],
            dim=1,
        )
        normals = head_outputs[..., :3]
        if self.normalize_normals:
            normals = F.normalize(normals, dim=-1)

        confidence = torch.sigmoid(head_outputs[..., 3:4])
        centers = center.unsqueeze(1).expand(
            -1,
            self.amount_of_plane_normals_predicted,
            -1,
        )

        return torch.cat((normals, centers, confidence), dim=-1)
