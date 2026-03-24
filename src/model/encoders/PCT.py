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
        # Normalizamos espacialmente usando softmax
        weights = F.softmax(scores, dim=2) # (B, 1, N)
        # Multiplicamos cada feature por su peso y sumamos a lo largo de los puntos
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
    def __init__(self, input_channels=3, num_points=1024, M_symmetries=8, hidden_dim=128, num_oa_layers=4):
        super().__init__()
        self.num_points = num_points
        self.M = M_symmetries

        # 1. Input Embedding
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=1) # <-- Usamos hidden_dim
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(hidden_dim) # <-- Usamos hidden_dim

        # 2. Encoder PCT (Offset Attention) DINÁMICO
        # Usamos nn.ModuleList para crear un número dinámico de capas OA.
        # Es OBLIGATORIO usar ModuleList en vez de una lista normal de Python [], 
        # para que PyTorch sepa que estas capas existen y deba entrenarlas.
        self.oa_layers = nn.ModuleList([OA(hidden_dim) for _ in range(num_oa_layers)])

        # 3. Spatially Weighted Pooling
        # Recibe la salida dinámica del Transformer
        self.sw_pooling = SpatiallyWeightedPooling(hidden_dim)

        # Dimensiones para la etapa de concatenación
        # Feature local + Feature global
        concat_dim = hidden_dim * 2

        # 4. Heads (Cabezales de Predicción - Decoders)
        # (Se mantienen iguales, ya que dependen de concat_dim que calculamos arriba)
        self.normal_head = nn.Sequential(
            nn.Conv1d(concat_dim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, M_symmetries * 3, 1) 
        )

        self.conf_head = nn.Sequential(
            nn.Conv1d(concat_dim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, M_symmetries, 1),
            nn.Sigmoid() 
        )

        self.center_head = nn.Sequential(
            nn.Conv1d(concat_dim, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # --- Etapa 1: Embedding Point-wise ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # --- Etapa 2: Transformer Dinámico ---
        # Pasamos la salida por cada una de las capas OA que pedimos
        for oa in self.oa_layers:
            x = oa(x)
            
        point_features = x # Salida: (B, hidden_dim, N)

        # --- Etapa 3: Global Feature Extraction (Spatially Weighted Pooling) ---
        global_feature, spatial_weights = self.sw_pooling(point_features)
        
        # Expandimos el vector global
        global_feature_expanded = global_feature.unsqueeze(-1).expand(-1, -1, num_points)

        # Concatenamos features locales y globales
        combined_features = torch.cat([point_features, global_feature_expanded], dim=1)

        # --- Etapa 4: Predicción (Heads) ---
        pred_normals = self.normal_head(combined_features) 
        pred_normals = pred_normals.permute(0, 2, 1).view(batch_size, num_points, self.M, 3)
        pred_normals = F.normalize(pred_normals, dim=-1)

        pred_confs = self.conf_head(combined_features) 
        pred_confs = pred_confs.permute(0, 2, 1) 

        pred_center = self.center_head(combined_features) 
        pred_center = pred_center.permute(0, 2, 1)

        return pred_normals, pred_confs, pred_center