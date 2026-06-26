import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatiallyWeightedPooling(nn.Module):
    """
    Spatially Weighted Pooling Module.
    Aprende a darle una ponderación (score) a cada punto de la nube
    para crear un vector global más representativo que un simple Max Pooling.
    """
    def __init__(self, channels):
        super(SpatiallyWeightedPooling, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(),
            nn.Conv1d(channels // 2, 1, 1)
        )

    def forward(self, x):
        # x shape: (B, channels, N)
        scores = self.mlp(x) # (B, 1, N)
        weights = F.softmax(scores, dim=2) # (B, 1, N)
        global_feat = torch.sum(x * weights, dim=2) # (B, channels)
        return global_feat, weights


class OA(nn.Module):
    """
    Offset-Attention Module.
    """
    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x


class PCT(nn.Module):
    """
    PCT Encoder puro. Extrae características densas (Point-wise + Global).
    No realiza predicciones de simetría.
    """
    def __init__(self, input_channels=3, hidden_dim=128, num_oa_layers=4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1. Input Embedding
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 2. Transformer Dinámico (Offset Attention)
        self.oa_layers = nn.ModuleList([OA(hidden_dim) for _ in range(num_oa_layers)])

        # 3. Spatially Weighted Pooling
        self.sw_pooling = SpatiallyWeightedPooling(hidden_dim)

        # La dimensión de salida será (hidden_dim * 2) porque concatenamos local y global
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        num_points = x.size(2)

        # --- Etapa 1: Embedding Point-wise ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # --- Etapa 2: Transformer ---
        for oa in self.oa_layers:
            x = oa(x)
            
        point_features = x # (B, hidden_dim, N)

        # --- Etapa 3: Global Feature Extraction ---
        global_feature, spatial_weights = self.sw_pooling(point_features)
        
        global_feature_expanded = global_feature.unsqueeze(-1).expand(-1, -1, num_points)

        # Concatenamos features locales y globales
        # Salida final: (B, hidden_dim*2, N)
        combined_features = torch.cat([point_features, global_feature_expanded], dim=1)

        # Retorna puramente los features extraídos
        return combined_features