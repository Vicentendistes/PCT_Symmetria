import torch
import torch.nn as nn
import torch.nn.functional as F

class SymPointLoss(nn.Module):
    def __init__(self, w_conf=1.0, w_vec=1.0, w_cent=1.0, w_rsd=0.1):
        super().__init__()
        # Pesos basados en los hiperparámetros del paper Symmetria:
        # alpha=1.0 (vec), beta=1.0 (cent), delta=1.0 (conf), gamma=0.1 (RSD)
        self.w_conf = w_conf
        self.w_vec = w_vec
        self.w_cent = w_cent
        self.w_rsd = w_rsd 
        
        # Loss para clasificación (Confianza): Binary Cross Entropy
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, points, pred_normals, pred_confs, pred_centers, gt_normals_list, gt_centers):
        """
        Calcula la Loss densa según Método M1 + RSD.
        points: (B, N, 3) -> Nube de puntos original (NECESARIO PARA RSD)
        gt_normals_list: Lista de tensores (K, 6) -> [Normal(3) + Punto en el plano(3)]
        """
        batch_size = pred_normals.shape[0]
        num_points = pred_normals.shape[1]
        
        total_loss = 0.0
        
        for b in range(batch_size):
            # --- Datos del objeto b ---
            P_obj = points[b]        # (N, 3) -> Nube de puntos
            P_n = pred_normals[b]    # (N, M, 3)
            P_c = pred_confs[b]      # (N, M)
            P_cent = pred_centers[b] # (N, 3)
            
            G_raw = gt_normals_list[b] # (K, 6)
            
            if G_raw is None or G_raw.shape[0] == 0:
                # Caso sin simetrías: penalizar confianza fuertemente
                zeros_target = torch.zeros_like(P_c)
                loss_conf = self.bce_loss(P_c, zeros_target).mean()
                total_loss += self.w_conf * loss_conf
                continue
                
            # Ground Truths: Normales y un Punto que pertenece a ese plano
            G_n = G_raw[:, 0:3]      # (K, 3) -> Vector Normal
            G_p = G_raw[:, 3:6]      # (K, 3) -> Punto en el plano
            G_cent_obj = gt_centers[b]   # (3) -> Centro de masa del objeto
            
            K = G_n.shape[0]
            
            # ============================================================
            # PASO 1: MATCHING (Asignar Predicciones a Ground Truth)
            # ============================================================
            P_n_norm = F.normalize(P_n, dim=-1)
            G_n_norm = F.normalize(G_n, dim=-1)
            
            dot_products = torch.matmul(P_n_norm, G_n_norm.t()) # (N, M, K)
            abs_dot = torch.abs(dot_products)
            
            best_heads_vals, best_heads_idx = torch.max(abs_dot, dim=1) # (N, K)
            
            # ============================================================
            # Eq (2): Confidence Loss (L_conf)
            # ============================================================
            target_conf = torch.zeros_like(P_c) 
            target_conf.scatter_(1, best_heads_idx, 1.0)
            loss_conf = self.bce_loss(P_c, target_conf).mean()
            
            # ============================================================
            # Eq (3): Vector Loss (L_vec)
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
            # NUEVO: Eq (4/5) del Paper - Reflection Symmetry Distance (L_RSD)
            # ============================================================
            # 1. Reflejar la nube usando el Plano Ground Truth (Perfecto)
            # Distancia de los puntos al plano GT: (P - PuntoPlano) dot Normal
            d_true = torch.sum((P_obj.unsqueeze(0) - G_p.unsqueeze(1)) * G_n_norm.unsqueeze(1), dim=2) # (K, N)
            P_ref_true = P_obj.unsqueeze(0) - 2 * d_true.unsqueeze(2) * G_n_norm.unsqueeze(1) # (K, N, 3)

            # 2. Reflejar la nube usando el Plano Predicho (Por cada punto)
            # Extraemos las normales predichas que hicieron match: (K, N, 3)
            n_pred = torch.stack([P_n_norm[torch.arange(num_points), best_heads_idx[:, k]] for k in range(K)])
            
            # Vector apuntando desde el centro predicho (P_cent) a cada punto de la nube (P_obj)
            # diff[v, u] es el vector desde el centro predicho 'v' al punto de la nube 'u'
            diff = P_obj.unsqueeze(0) - P_cent.unsqueeze(1) # (N_predictores, N_puntos, 3)
            
            # Distancia de los puntos a los planos predichos
            d_pred = torch.sum(diff.unsqueeze(0) * n_pred.unsqueeze(2), dim=-1) # (K, N_predictores, N_puntos)
            
            # Nubes reflejadas usando las predicciones: (K, N_predictores, N_puntos, 3)
            P_ref_pred = P_obj.unsqueeze(0).unsqueeze(0) - 2 * d_pred.unsqueeze(-1) * n_pred.unsqueeze(2)

            # 3. Calcular la diferencia (L1 Loss / Absolute Error por coordenada)
            P_ref_true_exp = P_ref_true.unsqueeze(1) # Expandir para comparar con los N predictores
            
            # Calculamos la media del error absoluto en los 3 ejes, promediado por todos los K planos y N puntos
            loss_rsd = F.l1_loss(P_ref_pred, P_ref_true_exp.expand_as(P_ref_pred))

            # ============================================================
            # Suma Total Ponderada
            # ============================================================
            loss_batch = (self.w_conf * loss_conf) + \
                         (self.w_vec * loss_vec) + \
                         (self.w_cent * loss_cent) + \
                         (self.w_rsd * loss_rsd)
                         
            total_loss += loss_batch

        return total_loss / batch_size