import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class SymPointLoss(nn.Module):
    def __init__(self, w_conf=1.0, w_vec=1.0, w_cent=1.0, w_rsd=0.3, conf_from_logits: bool = False):
        super().__init__()
        self.w_conf = w_conf
        self.w_vec = w_vec
        self.w_cent = w_cent
        self.w_rsd = w_rsd 
        self.conf_from_logits = conf_from_logits
        
        # Usaremos F.binary_cross_entropy directamente en el forward 
        # para inyectarle los pesos dinámicos de la asimetría de clases.

    def _confidence_loss(self, pred_confs, target_conf, weight=None):
        if self.conf_from_logits:
            return F.binary_cross_entropy_with_logits(pred_confs, target_conf, weight=weight)
        return F.binary_cross_entropy(pred_confs, target_conf, weight=weight)

    def forward(self, points, pred_normals, pred_confs, pred_centers, gt_normals_list, gt_centers):
        """
        Calcula la Loss densa optimizada con Bipartite Matching (Algoritmo Húngaro).
        """
        batch_size = pred_normals.shape[0]
        num_points = pred_normals.shape[1]
        M = pred_normals.shape[2] # Cantidad total de cabezales (ej. 32)
        
        total_loss = 0.0
        
        for b in range(batch_size):
            P_obj = points[b]        # (N, 3)
            P_n = pred_normals[b]    # (N, M, 3)
            P_c = pred_confs[b]      # (N, M)
            P_cent = pred_centers[b] # (N, 3)
            
            G_raw = gt_normals_list[b] 
            
            # Caso sin Ground Truths (e.g., sólidos de revolución pura)
            if G_raw is None or G_raw.shape[0] == 0:
                zeros_target = torch.zeros(P_c.shape, dtype=P_c.dtype, device=P_c.device)
                loss_conf = self._confidence_loss(P_c, zeros_target)
                total_loss += self.w_conf * loss_conf
                continue
                
            G_n = G_raw[:, 0:3]          # (K, 3) -> Vector Normal
            G_p = G_raw[:, 3:6]          # (K, 3) -> Punto en el plano
            G_cent_obj = gt_centers[b]   # (3) -> Centro de masa del objeto
            K = G_n.shape[0]
            if K > M:
                raise ValueError(
                    f"Got {K} ground-truth planes but only {M} prediction heads. "
                    "Increase amount_of_plane_normals_predicted or filter the labels."
                )
            
            P_n_norm = F.normalize(P_n, dim=-1)
            G_n_norm = F.normalize(G_n, dim=-1)
            
            # ============================================================
            # PASO 1: MATRIZ DE COSTOS MULTI-CRITERIO Y HUNGARIAN MATCHING
            # ============================================================
            # 1. Costo Angular
            dot_products = torch.matmul(P_n_norm, G_n_norm.t()) # (N, M, K)
            abs_dot = torch.abs(dot_products)
            cost_angle = (1.0 - abs_dot).mean(dim=0) # (M, K)
            
            # 2. Costo de Offset del Plano (Distancia ortogonal al origen)
            # Offset predictivo = promediamos (P_cent * P_n) sobre los N puntos -> (M)
            offset_pred = torch.sum(P_cent.unsqueeze(1) * P_n_norm, dim=-1).mean(dim=0) 
            # Offset GT = G_p * G_n -> (K)
            offset_gt = torch.sum(G_p * G_n_norm, dim=-1) 
            
            cost_offset = torch.abs(offset_pred.unsqueeze(1) - offset_gt.unsqueeze(0)) # (M, K)
            
            # 3. Matriz de Costos Final (Ángulo manda, Offset desempata. SIN confianza)
            cost_matrix = cost_angle + (0.5 * cost_offset)
            
            # Algoritmo Húngaro (requiere pasar a CPU temporalmente)
            C = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C)
            
            # Reordenamos los índices predictivos para que calcen exactamente con G_n (0 a K-1)
            sorted_match_idx = np.argsort(col_ind)
            matched_head_indices = row_ind[sorted_match_idx]

            # ============================================================
            # Eq (2): Confidence Loss (L_conf) - CON PESOS DINÁMICOS
            # ============================================================
            # Instanciamos tensor limpio para no heredar grafos computacionales
            target_conf = torch.zeros(P_c.shape, dtype=P_c.dtype, device=P_c.device)
            target_conf[:, matched_head_indices] = 1.0
            
            # Dinámica de pesos para balancear la asimetría masiva de clases
            h = float(M)
            r = float(K)
            p1 = (h - r) / h
            p2 = r / h
            weights = (p1 / r) * target_conf + (1.0 - target_conf) * (p2 / h)
            
            loss_conf = self._confidence_loss(P_c, target_conf, weight=weights)
            
            # ============================================================
            # Eq (3): Vector Loss (L_vec)
            # ============================================================
            # Extraemos solo los cabezales ganadores. Forma: (N, K, 3)
            P_n_matched = P_n_norm[:, matched_head_indices, :] 
            
            # Producto punto exacto 1 a 1:
            dot_matched = torch.sum(P_n_matched * G_n_norm.unsqueeze(0), dim=-1) # (N, K)
            
            cos_squared = dot_matched ** 2 
            vec_error = F.relu(1.0 - cos_squared)
            loss_vec = vec_error.mean() 
            
            # ============================================================
            # Eq (4): Center Loss (L_cent) - CON NORMA L1
            # ============================================================
            G_cent_expanded = G_cent_obj.unsqueeze(0).expand(num_points, 3)
            loss_cent = torch.norm(P_cent - G_cent_expanded, p=1, dim=1).mean()

            # ============================================================
            # Eq (5): Reflection Symmetry Distance (L_RSD) - VECTORIZADO
            # ============================================================
            # 1. Reflejo GT (Perfecto)
            d_true = torch.sum((P_obj.unsqueeze(1) - G_p.unsqueeze(0)) * G_n_norm.unsqueeze(0), dim=-1) # (N, K)
            P_ref_true = P_obj.unsqueeze(1) - 2 * d_true.unsqueeze(-1) * G_n_norm.unsqueeze(0) # (N, K, 3)

            # 2. Reflejo Predicho (Usando solo los cabezales emparejados)
            diff_pred = P_obj - P_cent # (N, 3)
            d_pred = torch.sum(diff_pred.unsqueeze(1) * P_n_matched, dim=-1) # (N, K)
            P_ref_pred = P_obj.unsqueeze(1) - 2 * d_pred.unsqueeze(-1) * P_n_matched # (N, K, 3)

            # 3. L1 Loss directo (Las formas ya cuadran perfecto)
            loss_rsd = F.l1_loss(P_ref_pred, P_ref_true)

            # ============================================================
            # Suma Total Ponderada
            # ============================================================
            loss_batch = (self.w_conf * loss_conf) + \
                         (self.w_vec * loss_vec) + \
                         (self.w_cent * loss_cent) + \
                         (self.w_rsd * loss_rsd)
                         
            total_loss += loss_batch

        return total_loss / batch_size
