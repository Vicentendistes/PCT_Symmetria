import torch
import torch.nn as nn


class InputTNet(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        self.point_mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.matrix_head = nn.Linear(256, k * k)

        # Así A comienza exactamente como la identidad.
        nn.init.zeros_(self.matrix_head.weight)
        nn.init.zeros_(self.matrix_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        features = self.point_mlp(x)           # (B, 1024, N)
        global_feature = features.max(dim=2).values
        global_feature = self.global_mlp(global_feature)

        delta = self.matrix_head(global_feature)
        delta = delta.reshape(batch_size, self.k, self.k)

        identity = torch.eye(
            self.k,
            dtype=x.dtype,
            device=x.device,
        ).unsqueeze(0)

        return identity + delta                # (B, 3, 3)


class DensePointNetEncoder(nn.Module):
    output_dim = 1088

    def __init__(
        self,
        input_channels: int = 3,
        use_input_tnet: bool = False,
    ):
        super().__init__()

        if use_input_tnet and input_channels != 3:
            raise ValueError(
                "Input T-Net solo transforma los tres canales XYZ."
            )

        self.use_input_tnet = use_input_tnet

        if use_input_tnet:
            self.input_tnet = InputTNet(k=3)

        self.local_mlp = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.deep_point_mlp = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, N)

        Returns:
            dense_features: (B, 1088, N)
            input_transform: (B, 3, 3) o None
        """
        batch_size, _, num_points = x.shape
        input_transform = None

        if self.use_input_tnet:
            input_transform = self.input_tnet(x)

            # P_aligned = P @ A
            x = torch.bmm(
                x.transpose(1, 2),
                input_transform,
            ).transpose(1, 2)

        local_features = self.local_mlp(x)
        # (B, 64, N)

        deep_features = self.deep_point_mlp(local_features)
        # (B, 1024, N)

        global_feature = deep_features.max(dim=2).values
        # (B, 1024)

        global_expanded = global_feature.unsqueeze(2).expand(
            -1, -1, num_points
        )
        # (B, 1024, N)

        dense_features = torch.cat(
            [local_features, global_expanded],
            dim=1,
        )
        # (B, 1088, N)

        return dense_features, input_transform