import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.encoders.PCT import SpatiallyWeightedPooling

class MHOA(nn.Module):
    """
    Multi-Head Offset-Attention Module con Pre-Layer Normalization.
    Actualizado a los estándares de estabilidad de LLMs modernos.
    """
    def __init__(self, channels, num_heads=4):
        super(MHOA, self).__init__()
        self.num_heads = num_heads
        
        # Validamos que los canales se puedan dividir en partes iguales
        assert channels % num_heads == 0, f"Los canales ({channels}) deben ser divisibles por los cabezales ({num_heads})"
        self.head_dim = channels // num_heads

        # 1. Pre-Normalization (El secreto para que no colapse al ser profundo)
        self.pre_norm = nn.BatchNorm1d(channels)

        # 2. Proyecciones Q, K, V
        # Mantenemos la reducción de dimensiones del PCT original para K y Q (channels // 4)
        # pero ahora dividido por el número de cabezales.
        self.k_dim = (channels // 4) // num_heads
        self.v_dim = channels // num_heads

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight # Peso compartido (mejora estabilidad en nubes de puntos)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        # 3. Feed Forward final (Proyección de salida)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, N = x.shape
        
        # --- PRE-NORM ---
        x_norm = self.pre_norm(x)

        # --- MULTI-HEAD PROJECTIONS ---
        # Proyectamos y remodelamos a: (Batch, Heads, Head_Dim, N)
        x_q = self.q_conv(x_norm).view(B, self.num_heads, self.k_dim, N).permute(0, 1, 3, 2) # (B, H, N, k_dim)
        x_k = self.k_conv(x_norm).view(B, self.num_heads, self.k_dim, N)                     # (B, H, k_dim, N)
        x_v = self.v_conv(x_norm).view(B, self.num_heads, self.v_dim, N)                     # (B, H, v_dim, N)

        # --- ATTENTION SCORE ---
        # Multiplicación de matrices por cabezal
        energy = torch.matmul(x_q, x_k) # (B, H, N, N)
        attention = self.softmax(energy)
        
        # Normalización Offset (L1-norm sobre la dimensión de puntos)
        attention = attention / (1e-9 + attention.sum(dim=2, keepdim=True))

        # --- APLICAR ATENCIÓN ---
        x_r = torch.matmul(x_v, attention) # (B, H, v_dim, N)
        
        # Volvemos a concatenar los cabezales: (B, C, N)
        x_r = x_r.reshape(B, C, N)

        # --- OFFSET Y CONEXIÓN RESIDUAL ---
        # La diferencia principal de Offset-Attention: input_norm - attention_out
        x_r = self.act(self.trans_conv(x_norm - x_r))
        
        # Residual connection tradicional
        x = x + x_r

        return x


    


class PCT_MHA(nn.Module):
    """
    PCT Encoder potenciado con Multi-Head Attention.
    """
    def __init__(self, input_channels=3, hidden_dim=128, num_oa_layers=4, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Instanciamos nuestras nuevas capas Multi-Head
        self.oa_layers = nn.ModuleList([MHOA(hidden_dim, num_heads=num_heads) for _ in range(num_oa_layers)])
        
        self.sw_pooling = SpatiallyWeightedPooling(hidden_dim)

    def forward(self, x):
        num_points = x.size(2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Transformer
        for oa in self.oa_layers:
            x = oa(x)
            
        point_features = x 
        global_feature, _ = self.sw_pooling(point_features)
        global_feature_expanded = global_feature.unsqueeze(-1).expand(-1, -1, num_points)
        
        combined_features = torch.cat([point_features, global_feature_expanded], dim=1)

        return combined_features