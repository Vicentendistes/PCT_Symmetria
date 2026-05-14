import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseSymmetryNet(nn.Module):
    def __init__(self, 
                 encoder_type="PCT", 
                 input_channels=3, 
                 M_symmetries=8, 
                 hidden_dim=128, 
                 num_oa_layers=4,
                 num_heads=4,
                 mha_attention_mode: str = "legacy",
                 mha_share_qk: bool = True,
                 mha_attn_dropout: float = 0.0,
                 mha_ffn_dropout: float = 0.0,
                 mha_norm_type: str = "instance",
                 mha_norm_affine: bool = False,
                 mha_residual_scale: float = 1.0):
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

        elif encoder_type == "PCT_MultiScale":
            from src.model.encoders.PCT_MultiScale import PCT_MultiScale
            self.encoder = PCT_MultiScale(input_channels, hidden_dim, num_oa_layers)
            # La dimensión final es: (128 * 4) locales + (128 * 4) globales = 1024
            self.encoder_output_dim = (hidden_dim * num_oa_layers) * 2


        elif encoder_type == "PCT_MHA":
            from src.model.encoders.PCT_MHA import PCT_MHA
            self.encoder = PCT_MHA(
                input_channels,
                hidden_dim,
                num_oa_layers,
                num_heads,
                attention_mode=mha_attention_mode,
                share_qk=mha_share_qk,
                attn_dropout=mha_attn_dropout,
                ffn_dropout=mha_ffn_dropout,
                norm_type=mha_norm_type,
                norm_affine=mha_norm_affine,
                residual_scale=mha_residual_scale,
            )
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

       # --- Etapa 3: Deshacer Rotación ---
        if trans_matrix is not None:
            inv_matrix = torch.linalg.inv(trans_matrix) 
            pred_center = torch.bmm(pred_center, inv_matrix)
            
            # ¡CAMBIO AQUÍ! Usamos reshape en vez de view
            normal_trans_matrix = trans_matrix.transpose(1, 2) # Transpuesta de la matriz original
            pred_normals_flat = torch.bmm(pred_normals_flat, normal_trans_matrix)
            
            # ¡CAMBIO AQUÍ TAMBIÉN!
            pred_normals = pred_normals_flat.reshape(batch_size, num_points, self.M, 3)
            pred_normals = F.normalize(pred_normals, dim=-1) 

        return pred_normals, pred_confs, pred_center
