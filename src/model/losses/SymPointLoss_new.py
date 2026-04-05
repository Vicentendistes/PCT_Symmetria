import torch
import torch.nn as nn
import torch.nn.functional as F

class SymPointLoss(nn.Module):
    def __init__(self, w_conf=1.0, w_vec=1.0, w_cent=1.0, w_rsd=0.1):
        super().__init__()
        # Pesos del paper Symmetria
        self.w_conf = w_conf
        self.w_vec = w_vec
        self.w_cent = w_cent
        self.w_rsd = w_rsd 
        
        # Loss para clasificación (Confianza): Binary Cross Entropy
        # Usamos reduction='none' para aplicar máscaras manualmente
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, points, pred_normals, pred_confs, pred_centers, gt_normals_list, gt_centers):
        """
        Calcula la Loss densa (Vectorizada) según Método M1 + RSD.
        """
        batch_size = pred_normals.shape[0]
        num_points = pred_normals.shape[1]
        device = pred_normals.device
        
        # ============================================================
        # PASO 0: PADDING DINÁMICO Y CREACIÓN DE MÁSCARA
        # ============================================================
        # Convertimos los Nones a tensores vacíos para poder empaquetarlos
        clean_list = [t if t is not None and t.numel() > 0 else torch.empty((0, 6), device=device) for t in gt_normals_list]
        
        # Hacemos padding de la lista: Todos tendrán el tamaño del objeto con más simetrías (K_max)
        # G_raw_padded shape: (B, K_max, 6)
        G_raw_padded = torch.nn.utils.rnn.pad_sequence(clean_list, batch_first=True) 
        K_max = G_raw_padded.shape[1]
        
        # Si no hay NINGUNA simetría en todo el batch (caso raro pero posible):
        if K_max == 0:
            loss_conf = self.bce_loss(pred_confs, torch.zeros_like(pred_confs)).mean()
            return self.w_conf * loss_conf
            
        # Máscara booleana para saber qué simetrías son reales y cuáles son relleno (padding)
        # valid_mask shape: (B, K_max)
        k_counts = torch.tensor([t.shape[0] for t in clean_list], device=device).unsqueeze(1)
        valid_mask = torch.arange(K_max, device=device).expand(batch_size, K_max) < k_counts

        # Extraemos Ground Truths
        G_n_norm = F.normalize(G_raw_padded[..., 0:3], dim=-1) # (B, K_max, 3)
        G_p = G_raw_padded[..., 3:6]                           # (B, K_max, 3)
        P_n_norm = F.normalize(pred_normals, dim=-1)           # (B, N, M, 3)

        # ============================================================
        # PASO 1: MATCHING (Asignar Predicciones a Ground Truth)
        # ============================================================
        # Producto punto usando einsum: (B, N, M, 3) dot (B, K_max, 3) -> (B, N, M, K_max)
        dot_products = torch.einsum('bnmc,bkc->bnmk', P_n_norm, G_n_norm)
        abs_dot = torch.abs(dot_products)
        
        # Ignoramos los valores acolchados llenándolos con -1 para que el max no los elija
        abs_dot = abs_dot.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(1), -1.0)
        
        # Encontramos la mejor cabeza (M) para cada Ground Truth (K) y cada Punto (N)
        # best_heads_vals, best_heads_idx shape: (B, N, K_max)
        best_heads_vals, best_heads_idx = torch.max(abs_dot, dim=2) 

        # ============================================================
        # Eq (2): Confidence Loss (L_conf)
        # ============================================================
        target_conf = torch.zeros_like(pred_confs) # (B, N, M)
        
        # Creamos un tensor de unos y ceros basado en la máscara
        src_mask = valid_mask.view(batch_size, 1, K_max).expand(batch_size, num_points, K_max).float()
        
        # Esparcimos los "unos" en las cabezas que hicieron match, ignorando los paddings
        target_conf.scatter_add_(2, best_heads_idx, src_mask)
        target_conf = target_conf.clamp_max(1.0) # Por si dos GT caen en la misma cabeza
        
        loss_conf = self.bce_loss(pred_confs, target_conf).mean()

        # ============================================================
        # Eq (3): Vector Loss (L_vec)
        # ============================================================
        vec_error = F.relu(1.0 - (best_heads_vals ** 2)) # (B, N, K_max)
        
        # Enmascaramos los errores de las simetrías falsas (padding)
        vec_error = vec_error * valid_mask.unsqueeze(1).float()
        
        # Promediamos solo sobre los elementos válidos
        valid_elements = valid_mask.sum() * num_points
        loss_vec = vec_error.sum() / (valid_elements + 1e-6)

        # ============================================================
        # Eq (4): Center Loss (L_cent)
        # ============================================================
        # Asumimos que los objetos sin simetrías no deben penalizar el centro
        has_sym_mask = (valid_mask.sum(dim=1) > 0).float() # (B,)
        
        loss_cent_raw = F.mse_loss(pred_centers, gt_centers.unsqueeze(1).expand(-1, num_points, -1), reduction='none') # (B, N, 3)
        loss_cent = (loss_cent_raw.mean(dim=(1,2)) * has_sym_mask).sum() / (has_sym_mask.sum() + 1e-6)

        # ============================================================
        # Eq (4/5) del Paper - Reflection Symmetry Distance (L_RSD)
        # ============================================================
        # 1. Reflejo Verdadero (True Reflection)
        # diff_true: (B, K_max, N, 3). Vector desde el punto en el plano a los puntos de la nube
        diff_true = points.unsqueeze(1) - G_p.unsqueeze(2) 
        d_true = torch.einsum('bkni,bki->bkn', diff_true, G_n_norm)
        P_ref_true = points.unsqueeze(1) - 2 * d_true.unsqueeze(-1) * G_n_norm.unsqueeze(2) # (B, K_max, N, 3)

        # 2. Reflejo Predicho (Predicted Reflection)
        # Recolectamos las normales predichas que hicieron match: (B, K_max, N_predictores, 3)
        idx = best_heads_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        n_pred = torch.gather(P_n_norm, 2, idx).permute(0, 2, 1, 3) 
        
        # Distancia a los planos predichos optimizada con einsum
        term1 = torch.einsum('bui,bkvi->bkuv', points, n_pred)        # (B, K_max, N_obj, N_pred)
        term2 = torch.einsum('bvi,bkvi->bkv', pred_centers, n_pred)   # (B, K_max, N_pred)
        d_pred = term1 - term2.unsqueeze(2)                           # (B, K_max, N_obj, N_pred)

        # Calculamos la reflexión de la nube para todos los predictores a la vez
        # Shape: (Batch, K_max, N_obj, N_pred, 3)
        P_ref_pred = points.view(batch_size, 1, num_points, 1, 3) - \
                     2 * d_pred.unsqueeze(-1) * n_pred.unsqueeze(2)

        # 3. Diferencia L1 Masked
        P_ref_true_exp = P_ref_true.unsqueeze(3) # Para hacer broadcast contra los N predictores
        loss_rsd_matrix = F.l1_loss(P_ref_pred, P_ref_true_exp.expand_as(P_ref_pred), reduction='none')
        loss_rsd_per_k = loss_rsd_matrix.mean(dim=(2, 3, 4)) # (B, K_max)

        # Aplicamos la máscara booleana para matar las losses de los planos de padding
        loss_rsd_per_k = loss_rsd_per_k * valid_mask.float()
        loss_rsd = loss_rsd_per_k.sum() / (valid_mask.sum() + 1e-6)

        # ============================================================
        # Suma Total Ponderada
        # ============================================================
        total_loss = (self.w_conf * loss_conf) + \
                     (self.w_vec * loss_vec) + \
                     (self.w_cent * loss_cent) + \
                     (self.w_rsd * loss_rsd)
                     
        return total_loss