import torch
from lightning.pytorch.cli import LightningCLI
from src.dataset.SymDataModule import SymDataModule

# IMPORTAMOS EL NUEVO MODELO
from src.model.LightningSymmetryModel import LightningSymmetryModel

def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(
        model_class=LightningSymmetryModel,  # <-- CAMBIO APLICADO AQUÍ
        datamodule_class=SymDataModule,
        save_config_callback=None
    )

if __name__ == "__main__":
    cli_main()