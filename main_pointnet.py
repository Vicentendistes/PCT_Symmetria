import torch
from lightning.pytorch.cli import LightningCLI

from src.dataset.SymDataModule import SymDataModule
from src.model.LightningSantelicesPointNet import (
    LightningSantelicesPointNet,
)


def cli_main() -> None:
    torch.set_float32_matmul_precision("high")
    LightningCLI(
        model_class=LightningSantelicesPointNet,
        datamodule_class=SymDataModule,
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()
