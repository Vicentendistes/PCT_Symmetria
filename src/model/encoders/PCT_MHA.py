import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.encoders.PCT import SpatiallyWeightedPooling


class MHOA(nn.Module):
    """
    Multi-Head Offset-Attention Module con Pre-Layer Normalization.
    Actualizado con FFN x4 y Escalado de Atención (\sqrt{d_k}).
    """
    def __init__(self, channels, num_heads=4):
        super(MHOA, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # 1. Pre-Norm: Usamos InstanceNorm1d en vez de BatchNorm para mayor estabilidad
        self.pre_norm_attn = nn.InstanceNorm1d(channels)
        self.pre_norm_ffn = nn.InstanceNorm1d(channels)

        # 2. Proyecciones Multi-Head (Mantenemos la lógica de pesos compartidos para Q y K)
        self.k_dim = (channels // 4) // num_heads
        self.v_dim = channels // num_heads

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

        # 3. Proyección de salida de la atención
        self.attn_out_conv = nn.Conv1d(channels, channels, 1)

        # 4. NUEVO: Feed-Forward Network (Proporción Estándar 1:4)
        ffn_expansion = channels * 4  # <-- CAMBIO AQUÍ: Expansión x4 en vez de x2
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, ffn_expansion, 1),
            nn.GELU(), 
            nn.Conv1d(ffn_expansion, channels, 1)
        )

    def forward(self, x):
        B, C, N = x.shape
        
        # ==========================================
        # BLOQUE 1: MULTI-HEAD ATTENTION
        # ==========================================
        x_norm = self.pre_norm_attn(x)

        x_q = self.q_conv(x_norm).view(B, self.num_heads, self.k_dim, N).permute(0, 1, 3, 2)
        x_k = self.k_conv(x_norm).view(B, self.num_heads, self.k_dim, N)
        x_v = self.v_conv(x_norm).view(B, self.num_heads, self.v_dim, N)

        # <-- CAMBIO AQUÍ: Escalado por la raíz cuadrada de la dimensión del cabezal
        energy = torch.matmul(x_q, x_k) / (self.k_dim ** 0.5) 
        
        # <-- CAMBIO AQUÍ: Eliminamos la normalización L1 para evitar aplanar los pesos.
        # Ahora el Softmax escalado se encarga de todo.
        attention = self.softmax(energy) 

        x_r = torch.matmul(x_v, attention) 
        
        # ¡CORRECCIÓN CRÍTICA APLICADA!
        x_r = x_r.contiguous().reshape(B, C, N)

        # Offset y conexión residual (Offset = Input - Attn)
        attn_out = self.attn_out_conv(x_norm - x_r)
        x = x + attn_out # Residual 1

        # ==========================================
        # BLOQUE 2: FEED-FORWARD NETWORK
        # ==========================================
        x_ffn_norm = self.pre_norm_ffn(x)
        ffn_out = self.ffn(x_ffn_norm)
        x = x + ffn_out # Residual 2

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