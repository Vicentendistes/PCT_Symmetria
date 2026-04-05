import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.encoders.PCT import OA, SpatiallyWeightedPooling

class PCT_MultiScale(nn.Module):
    """
    PCT Encoder con Visión Multi-Escala.
    Concatena la salida de TODAS las capas de Offset-Attention antes del pooling
    para no perder detalles geométricos de bajo nivel.
    """
    def __init__(self, input_channels=3, hidden_dim=128, num_oa_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_oa_layers = num_oa_layers

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.oa_layers = nn.ModuleList([OA(hidden_dim) for _ in range(num_oa_layers)])
        
        # OJO AQUÍ: El Pooling ahora recibe un tensor mucho más gordo
        # Si hidden_dim=128 y num_oa_layers=4, recibirá 128 * 4 = 512 canales
        self.concat_dim = hidden_dim * num_oa_layers
        self.sw_pooling = SpatiallyWeightedPooling(self.concat_dim)

    def forward(self, x):
        num_points = x.size(2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # --- Etapa 2: Transformer Multi-Escala ---
        oa_outputs = []
        for oa in self.oa_layers:
            x = oa(x)
            oa_outputs.append(x)
            
        # Concatenamos a lo largo de los canales (dim=1)
        # Salida: (B, hidden_dim * num_oa_layers, N)
        point_features = torch.cat(oa_outputs, dim=1) 
        
        # --- Etapa 3: Pooling Global ---
        global_feature, _ = self.sw_pooling(point_features)
        global_feature_expanded = global_feature.unsqueeze(-1).expand(-1, -1, num_points)
        
        # Juntamos lo local multi-escala con lo global
        combined_features = torch.cat([point_features, global_feature_expanded], dim=1)

        return combined_features