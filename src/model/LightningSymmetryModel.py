import lightning
import torch
from src.model.DenseSymmetryNet import DenseSymmetryNet
from src.model.losses.SymPointLoss import SymPointLoss

class LightningSymmetryModel(lightning.LightningModule):
    def __init__(self,
                 encoder_type: str = "PCT",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 amount_of_plane_normals_predicted: int = 8,
                 w_conf: float = 1.0,
                 w_vec: float = 1.0,
                 w_cent: float = 1.0,
                 w_rsd: float = 0.1,
                 input_channels: int = 3,
                 hidden_dim: int = 128,
                 num_oa_layers: int = 4,
                 num_heads: int = 4,
                 mha_attention_mode: str = "legacy",
                 mha_share_qk: bool = True,
                 mha_attn_dropout: float = 0.0,
                 mha_ffn_dropout: float = 0.0,
                 mha_norm_type: str = "instance",
                 mha_norm_affine: bool = False,
                 mha_residual_scale: float = 1.0
                 ):
        """
        Lightning Module Universal para Predicción Densa de Simetrías.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 1. El Modelo Modular (Instancia el Cerebro y el Encoder)
        self.net = DenseSymmetryNet(
            encoder_type=encoder_type,
            input_channels=input_channels,
            M_symmetries=amount_of_plane_normals_predicted,
            hidden_dim=hidden_dim,
            num_oa_layers=num_oa_layers,
            num_heads=num_heads,
            mha_attention_mode=mha_attention_mode,
            mha_share_qk=mha_share_qk,
            mha_attn_dropout=mha_attn_dropout,
            mha_ffn_dropout=mha_ffn_dropout,
            mha_norm_type=mha_norm_type,
            mha_norm_affine=mha_norm_affine,
            mha_residual_scale=mha_residual_scale
        )
        
        # 2. La Función de Pérdida (M1 + RSD)
        self.loss_fn = SymPointLoss(
            w_conf=w_conf,
            w_vec=w_vec,
            w_cent=w_cent,
            w_rsd=w_rsd
        )

        self.lr = learning_rate

    def configure_optimizers(self):
        # 1. AdamW: El estándar para entrenar Transformers y evitar sobreajuste
        # Accedemos a weight_decay a través de self.hparams que Lightning guarda automáticamente
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.hparams.weight_decay 
        )
        
        # 2. CosineAnnealing: Baja el LR en forma de curva suave hasta llegar a eta_min
        # Usamos self.trainer.max_epochs para que la curva calce exacto con tu YAML
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs, 
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # Se actualiza al final de cada época
                "monitor": "val_loss" # Aunque Cosine no usa el monitor, es buena práctica dejarlo
            }
        }

    def forward(self, x):
        if x.shape[1] != 3:
             x = x.transpose(1, 2)
        return self.net(x)

    def _step(self, batch, step_tag):
        points_list = batch.get_points()
        points = torch.stack(points_list).to(self.device).float()
        
        gt_normals_raw = batch.get_plane_syms()
        gt_normals = [t.to(self.device).float() if t is not None else None for t in gt_normals_raw]
        
        gt_centers = torch.zeros((points.shape[0], 3), device=self.device)

        points_input = points.transpose(1, 2) 
        
        pred_n, pred_c, pred_cent = self.net(points_input)

        loss = self.loss_fn(
            points=points,
            pred_normals=pred_n,
            pred_confs=pred_c,
            pred_centers=pred_cent,
            gt_normals_list=gt_normals,
            gt_centers=gt_centers
        )

        self.log(f"{step_tag}_loss", loss, prog_bar=True, batch_size=points.shape[0], sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        points = torch.stack(batch.get_points()).float()
        points_input = points.transpose(1, 2)
        pred_n, pred_c, pred_cent = self.net(points_input)
        return batch, pred_n, pred_c, pred_cent

    def on_after_backward(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                if param.grad.isnan().any():
                    print(f"¡ALERTA! NaN detectado en gradientes de: {name}")
