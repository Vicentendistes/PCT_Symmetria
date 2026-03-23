import torch
from lightning.pytorch.cli import LightningCLI
from src.dataset.SymDataModule import SymDataModule
from src.model.LightningPCT import LightningPCT



def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(
        model_class=LightningPCT, 
        datamodule_class=SymDataModule,
        save_config_callback=None # Opcional: evita crear copias del config si solo estás probando
    )


if __name__ == "__main__":
    cli_main()
