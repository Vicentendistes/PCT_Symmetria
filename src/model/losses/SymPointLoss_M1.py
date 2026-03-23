import torch
import torch.nn as nn
import torch.nn.functional as F

class SymPointLoss_M1(nn.Module):
    def __init__(self, w_conf=1.0, w_vec=1.0, w_cent=1.0):
        super().__init__()
        self.w_conf = w_conf
        self.w_vec = w_vec
        self.w_cent = w_cent
        
        # Loss para clasificación (Confianza): Binary Cross Entropy
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, pred_normals, pred_confs, pred_centers, gt_normals_list, gt_centers):
        """
        Calcula la Loss densa según Método M1.
        gt_normals_list: Lista de tensores (K, 6) -> [Normal(3) + Punto(3)]
        """
        batch_size = pred_normals.shape[0]
        num_points = pred_normals.shape[1]
        
        total_loss = 0.0
        
        for b in range(batch_size):
            # --- Datos del objeto b ---
            P_n = pred_normals[b]    # (N, M, 3)
            P_c = pred_confs[b]      # (N, M)
            P_cent = pred_centers[b] # (N, 3)
            
            # --- CORRECCIÓN AQUÍ ---
            # El dataset entrega (K, 6). Extraemos solo las normales (primeras 3 cols)
            G_raw = gt_normals_list[b] # (K, 6)
            
            if G_raw is None or G_raw.shape[0] == 0:
                # Caso sin simetrías: penalizar confianza
                zeros_target = torch.zeros_like(P_c)
                loss_conf = self.bce_loss(P_c, zeros_target).mean()
                total_loss += self.w_conf * loss_conf
                continue
                
            # Tomamos solo las normales (nx, ny, nz)
            G_n = G_raw[:, 0:3]      # (K, 3)
            G_cent_obj = gt_centers[b]   # (3) -> Centro del objeto (usualmente 0,0,0)
            
            K = G_n.shape[0]
            
            # ============================================================
            # PASO 1: MATCHING (Asignar Predicciones a Ground Truth)
            # ============================================================
            
            # P_n_norm: (N, M, 3)
            P_n_norm = F.normalize(P_n, dim=-1)
            # G_n_norm: (K, 3)
            G_n_norm = F.normalize(G_n, dim=-1)
            
            # Para hacer matmul, expandimos para broadcasting
            # P_n_norm: (N, M, 3)
            # G_n_norm.t(): (3, K)
            # Resultado: (N, M, K) -> Para cada punto y cada cabeza, qué tan cerca está de cada GT
            # Ahora las dimensiones coinciden (3 con 3)
            dot_products = torch.matmul(P_n_norm, G_n_norm.t()) 
            abs_dot = torch.abs(dot_products)
            
            # best_heads_vals: (N, K) -> El mejor coseno para cada GT
            # best_heads_idx:  (N, K) -> Qué cabeza fue la mejor
            best_heads_vals, best_heads_idx = torch.max(abs_dot, dim=1) 
            
            # ============================================================
            # Eq (2): Confidence Loss (L_conf)
            # ============================================================
            target_conf = torch.zeros_like(P_c) 
            target_conf.scatter_(1, best_heads_idx, 1.0)
            
            loss_conf = self.bce_loss(P_c, target_conf).mean()
            
            # ============================================================
            # Eq (3): Vector Loss (L_vec)
            # max(0, 1 - |n . n_hat|^2)
            # ============================================================
            cos_squared = best_heads_vals ** 2 
            vec_error = F.relu(1.0 - cos_squared)
            loss_vec = vec_error.mean() 
            
            # ============================================================
            # Eq (4): Center Loss (L_cent)
            # ============================================================
            G_cent_expanded = G_cent_obj.unsqueeze(0).expand(num_points, 3)
            loss_cent = F.mse_loss(P_cent, G_cent_expanded)
            
            # ============================================================
            # Suma Total
            # ============================================================
            loss_batch = (self.w_conf * loss_conf) + \
                         (self.w_vec * loss_vec) + \
                         (self.w_cent * loss_cent)
                         
            total_loss += loss_batch

        return total_loss / batch_size