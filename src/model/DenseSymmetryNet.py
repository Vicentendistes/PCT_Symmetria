import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseSymmetryNet(nn.Module):
    def __init__(self, 
                 encoder_type="PCT", 
                 input_channels=3, 
                 M_symmetries=8, 
                 hidden_dim=128, 
                 num_oa_layers=4):
        super().__init__()
        self.M = M_symmetries
        self.encoder_type = encoder_type

        # ==========================================
        # 1. CARGA DINÁMICA DEL ENCODER
        # ==========================================
        if encoder_type == "PCT":
            from src.model.encoders.PCT import PCT
            self.encoder = PCT(input_channels, hidden_dim, num_oa_layers)
            self.encoder_output_dim = hidden_dim * 2
            
        elif encoder_type == "PCT_TNet":
            from src.model.encoders.PCT_TNet import PCT_TNet
            self.encoder = PCT_TNet(input_channels, hidden_dim, num_oa_layers)
            self.encoder_output_dim = hidden_dim * 2
            
        else:
            raise ValueError(f"Encoder '{encoder_type}' no está soportado.")

        # ==========================================
        # 2. CABEZALES DE PREDICCIÓN
        # ==========================================
        concat_dim = self.encoder_output_dim

        self.normal_head = nn.Sequential(
            nn.Conv1d(concat_dim, 512, 1), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.ReLU(),
            nn.Conv1d(256, self.M * 3, 1) 
        )
        self.conf_head = nn.Sequential(
            nn.Conv1d(concat_dim, 512, 1), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.ReLU(),
            nn.Conv1d(256, self.M, 1), nn.Sigmoid() 
        )
        self.center_head = nn.Sequential(
            nn.Conv1d(concat_dim, 256, 1), nn.ReLU(),
            nn.Conv1d(256, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        trans_matrix = None

        # --- Etapa 1: Extracción ---
        if self.encoder_type == "PCT_TNet":
            features, trans_matrix = self.encoder(x)
        else:
            features = self.encoder(x)

        # --- Etapa 2: Predicción ---
        pred_normals = self.normal_head(features) 
        pred_normals = pred_normals.permute(0, 2, 1).view(batch_size, num_points, self.M, 3)
        pred_normals = F.normalize(pred_normals, dim=-1)

        pred_confs = self.conf_head(features) 
        pred_confs = pred_confs.permute(0, 2, 1) 

        pred_center = self.center_head(features) 
        pred_center = pred_center.permute(0, 2, 1)

        # --- Etapa 3: Deshacer Rotación (Si existe T-Net) ---
        if trans_matrix is not None:
            # Calculamos la matriz inversa
            inv_matrix = torch.linalg.inv(trans_matrix) # (B, 3, 3)
            
            # Rotar los centros: (B, num_points, 3) * (B, 3, 3)
            pred_center = torch.bmm(pred_center, inv_matrix)
            
            # Rotar las normales: Aplanamos a (B, N*M, 3) para multiplicar fácilmente
            pred_normals_flat = pred_normals.view(batch_size, num_points * self.M, 3)
            pred_normals_flat = torch.bmm(pred_normals_flat, inv_matrix)
            pred_normals = pred_normals_flat.view(batch_size, num_points, self.M, 3)
            pred_normals = F.normalize(pred_normals, dim=-1) # Re-normalizar por seguridad

        return pred_normals, pred_confs, pred_center