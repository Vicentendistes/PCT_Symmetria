import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoders.PCT import OA, SpatiallyWeightedPooling

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        matrix = self.fc3(x).view(-1, self.k, self.k)
        # Inicialización con matriz identidad
        iden = torch.eye(self.k, dtype=matrix.dtype, device=matrix.device).view(1, self.k, self.k).repeat(B, 1, 1)
        matrix = matrix + iden
        return matrix

class PCT_TNet(nn.Module):
    """
    PCT con módulo T-Net integrado para invarianza espacial.
    """
    def __init__(self, input_channels=3, hidden_dim=128, num_oa_layers=4):
        super().__init__()
        self.t_net = TNet(k=3)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.oa_layers = nn.ModuleList([OA(hidden_dim) for _ in range(num_oa_layers)])
        self.sw_pooling = SpatiallyWeightedPooling(hidden_dim)

    def forward(self, x):
        num_points = x.size(2)

        # 1. Alineación Espacial (T-Net)
        trans_matrix = self.t_net(x) # (B, 3, 3)
        x_transposed = x.transpose(2, 1)
        x_aligned = torch.bmm(x_transposed, trans_matrix)
        x = x_aligned.transpose(2, 1)

        # 2. Extracción PCT (Sobre los puntos ya alineados)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        for oa in self.oa_layers:
            x = oa(x)
        point_features = x
        global_feature, _ = self.sw_pooling(point_features)
        global_feature_expanded = global_feature.unsqueeze(-1).expand(-1, -1, num_points)
        combined_features = torch.cat([point_features, global_feature_expanded], dim=1)

        # Retornamos las features Y la matriz para poder deshacer la rotación después
        return combined_features, trans_matrix