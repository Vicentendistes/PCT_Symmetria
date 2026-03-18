import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]

        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x



class PCT_M1(nn.Module):
    def __init__(self, input_channels=3, num_points=1024, M_symmetries=8):
        """
        Implementación de PCT según el modelo obtenido en SHREC 2023.
        """
        super().__init__()
        self.num_points = num_points
        self.M = M_symmetries

        # 1. Input Embedding
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        # 2. Encoder PCT (Offset Attention)
        self.oa1 = OA(128)
        self.oa2 = OA(128)
        self.oa3 = OA(128)
        self.oa4 = OA(128)

        # Dimensiones para la etapa de concatenación
        # Feature local (128) + Feature global (128) = 256
        concat_dim = 256

        # 3. Heads (Cabezales de Predicción - Decoders)
        # Estos son MLPs aplicados a cada punto (Conv1d con kernel 1 actúa como MLP point-wise)
        
        # Rama A: Normales (M vectores de 3 dimensiones) -> Salida: M * 3
        self.normal_head = nn.Sequential(
            nn.Conv1d(concat_dim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, M_symmetries * 3, 1) 
        )

        # Rama B: Confianza (M probabilidades) -> Salida: M
        self.conf_head = nn.Sequential(
            nn.Conv1d(concat_dim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, M_symmetries, 1),
            nn.Sigmoid() # Importante: Sigmoid fuerza salida entre 0 y 1
        )

        # Rama C: Centro (1 punto de 3 dimensiones) -> Salida: 3
        self.center_head = nn.Sequential(
            nn.Conv1d(concat_dim, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x):
        # x shape de entrada: (Batch, 3, N)
        # Nota: En Pytorch las convoluciones esperan (Batch, Canales, Largo/Puntos)
        
        batch_size = x.size(0)
        num_points = x.size(2)

        # --- Etapa 1: Embedding Point-wise ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # Salida: (B, 128, N)

        # --- Etapa 2: Transformer (Contexto Local-Global por Atención) ---
        x = self.oa1(x)
        x = self.oa2(x)
        x = self.oa3(x)
        point_features = self.oa4(x) # Salida: (B, 128, N)

        # --- Etapa 3: Global Feature Extraction & Concat ---
        # Extraemos el vector global (Max Pooling sobre la dimensión de puntos)
        # (B, 128, N) -> (B, 128, 1)
        global_feature = torch.max(point_features, dim=2, keepdim=True)[0]
        
        # Repetimos el vector global para cada punto
        # (B, 128, 1) -> (B, 128, N)
        global_feature_repeated = global_feature.repeat(1, 1, num_points)

        # Concatenamos features locales y globales
        # (B, 128, N) concatenado con (B, 128, N) -> (B, 256, N)
        combined_features = torch.cat([point_features, global_feature_repeated], dim=1)

        # --- Etapa 4: Predicción (Heads) ---
        
        # 1. Normales
        pred_normals = self.normal_head(combined_features) # (B, M*3, N)
        # Reorganizamos para tener (B, N, M, 3) que es más fácil de manejar
        pred_normals = pred_normals.permute(0, 2, 1).view(batch_size, num_points, self.M, 3)
        # Normalizamos los vectores para que tengan magnitud 1 (importante para cosenos)
        pred_normals = F.normalize(pred_normals, dim=-1)

        # 2. Confianza
        pred_confs = self.conf_head(combined_features) # (B, M, N)
        pred_confs = pred_confs.permute(0, 2, 1) # (B, N, M)

        # 3. Centro
        pred_center = self.center_head(combined_features) # (B, 3, N)
        pred_center = pred_center.permute(0, 2, 1) # (B, N, 3)

        return pred_normals, pred_confs, pred_center