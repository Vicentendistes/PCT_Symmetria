import lightning
import torch
from src.model.encoders.PCT_M1 import PCT_M1
from src.model.losses.SymPointLoss import SymPointLoss

class LightningPCT_M1(lightning.LightningModule):
    def __init__(self,
                 learning_rate: float = 1e-4,
                 amount_of_plane_normals_predicted: int = 8, # M del paper
                 w_conf: float = 1.0,
                 w_vec: float = 1.0,
                 w_cent: float = 1.0,
                 input_channels: int = 3,
                 n_points: int = 2048
                 ):
        """
        Lightning Module específico para el método M1 (Dense Prediction).
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 1. El Modelo (PCT Modificado para salida densa)
        self.net = PCT_M1(
            input_channels=input_channels,
            num_points=n_points,
            M_symmetries=amount_of_plane_normals_predicted
        )
        
        # 2. La Función de Pérdida (Ecuaciones 1-4 del Paper)
        self.loss_fn = SymPointLoss(
            w_conf=w_conf,
            w_vec=w_vec,
            w_cent=w_cent
        )

        self.lr = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Si el val_loss no mejora en 10 épocas, divide el LR por 2 (factor=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            min_lr=1e-5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" # Monitorea la misma métrica que el Early Stopping
            }
        }

    def forward(self, x):
        # x shape: (Batch, N, 3) -> Transpose -> (Batch, 3, N)
        if x.shape[1] != 3:
             x = x.transpose(1, 2)
        return self.net(x)

    def _step(self, batch, step_tag):
        # 1. Preparar Datos
        points_list = batch.get_points()
        
        # CORRECCIÓN: Asegurarnos que el stack esté en self.device (GPU)
        # batch.get_points() ya debería tener tensores en device si el batcher lo hizo bien,
        # pero para estar 100% seguros:
        points = torch.stack(points_list).to(self.device).float()
        
        # Ground Truths
        # batch.get_plane_syms() devuelve una lista de tensores. 
        # Asegúrate de mover cada tensor de la lista a la GPU también si no lo están.
        gt_normals_raw = batch.get_plane_syms()
        gt_normals = [t.to(self.device).float() if t is not None else None for t in gt_normals_raw]
        
        # GT Centers
        gt_centers = torch.zeros((points.shape[0], 3), device=self.device)

        # 2. Forward Pass
        # Transponemos entrada para PCT: (B, 3, N)
        points_input = points.transpose(1, 2) 
        
        pred_n, pred_c, pred_cent = self.net(points_input)

        # 3. Calcular Loss
        loss = self.loss_fn(
            pred_normals=pred_n,
            pred_confs=pred_c,
            pred_centers=pred_cent,
            gt_normals_list=gt_normals,
            gt_centers=gt_centers
        )

        # 4. Logging (Simplificado)
        self.log(f"{step_tag}_loss", loss, prog_bar=True, batch_size=points.shape[0], sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # En validación solo miramos la Loss por ahora.
        # Las métricas complejas (MAP) requieren post-procesamiento (DBSCAN) 
        # que es lento para ejecutar en cada epoch.
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Útil para inferencia o generar visualizaciones
        points = torch.stack(batch.get_points()).float()
        points_input = points.transpose(1, 2)
        pred_n, pred_c, pred_cent = self.net(points_input)
        return batch, pred_n, pred_c, pred_cent

    def on_after_backward(self):
        # BUENA PRÁCTICA: Chequeo de seguridad para gradientes explosivos/NaNs
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                if param.grad.isnan().any():
                    print(f"¡ALERTA! NaN detectado en gradientes de: {name}")