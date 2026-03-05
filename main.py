import torch
from lightning.pytorch.cli import LightningCLI
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from src.dataset.SymDataModule import SymDataModule
from src.model.LightningPCT_M1 import LightningPCT_M1



def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(
        model_class=LightningPCT_M1, 
        datamodule_class=SymDataModule,
        save_config_callback=None # Opcional: evita crear copias del config si solo estás probando
    )


if __name__ == "__main__":
    cli_main()
