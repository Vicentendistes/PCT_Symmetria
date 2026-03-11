import copy
import lzma
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm  # <--- NUEVO: Para ver la barra de carga en RAM

from src.dataset.SymDatasetItem import SymDatasetItem
from src.dataset.transforms.AbstractTransform import AbstractTransform
from src.dataset.transforms.ComposeTransform import ComposeTransform
from src.dataset.transforms.IdentityTransform import IdentityTransform
from src.dataset.transforms.RandomSampler import RandomSampler
from src.dataset.transforms.UnitSphereNormalization import UnitSphereNormalization

class SymDataset(Dataset):      
    def __init__(
            self,
            data_source_path: str = "path/to/dataset/split",
            transform: AbstractTransform = IdentityTransform(),
            has_ground_truth: bool = True,
            shape_excluded: Optional[List[str]] = None,
            perturbation_excluded: Optional[List[str]] = None,
            debug=False,
            load_to_ram: bool = False  # <--- NUEVO: Activador de memoria RAM
    ):
        self.data_source_path = Path(data_source_path)
        self.transform = transform
        self.has_ground_truth = has_ground_truth
        self.debug = debug
        self.load_to_ram = load_to_ram

        self.shape_excluded = [] if shape_excluded is None else shape_excluded
        self.perturbation_excluded = [] if perturbation_excluded is None else perturbation_excluded

        self.filename_list = list(self.data_source_path.rglob(f'*/*.xz'))
        self.filename_list = [
            filename for filename in self.filename_list
            if str(filename.name).split("-")[1] not in self.shape_excluded
               and str(filename.name).split("-")[2].replace(".xz", "") not in self.perturbation_excluded
        ]
        self.length = len(self.filename_list)

        # ==========================================
        # NUEVO BLOQUE: CARGA MASIVA A LA RAM
        # ==========================================
        self.ram_cache = []
        if self.load_to_ram:
            print(f"[{self.data_source_path.name}] Cargando {self.length} modelos en RAM...")
            # Leemos todos los archivos AHORA, una sola vez
            for i in tqdm(range(self.length), desc="Cargando a RAM"):
                points = self.read_points(i)
                if self.has_ground_truth:
                    planes = self.read_planes(i)
                else:
                    planes = (None, None, None)
                # Guardamos los tensores en la lista (memoria)
                self.ram_cache.append((points, planes))
            print(f"[{self.data_source_path.name}] ¡Carga en RAM completada!")

    def _parse_sym_file(self, filename):
        planar_symmetries = []
        axis_continue_symmetries = []
        axis_discrete_symmetries = []

        with open(filename) as f:
            line_amount = int(f.readline())
            for _ in range(line_amount):
                line = f.readline().split(" ")
                line = [x.replace("\n", "") for x in line]
                if line[0] == "plane":
                    plane = [float(x) for x in line[1::]]
                    planar_symmetries.append(torch.tensor(plane))
                elif line[0] == "axis" and line[-1] == "inf":
                    plane = [float(x) for x in line[1:7]]
                    axis_continue_symmetries.append(torch.tensor(plane))
                else:
                    plane = [float(x) for x in line[1::]]
                    axis_discrete_symmetries.append(torch.tensor(plane))

        planar_symmetries = None if len(planar_symmetries) == 0 else torch.stack(planar_symmetries).float()
        axis_continue_symmetries = None if len(axis_continue_symmetries) == 0 else torch.stack(
            axis_continue_symmetries).float()
        axis_discrete_symmetries = None if len(axis_discrete_symmetries) == 0 else torch.stack(
            axis_discrete_symmetries).float()
            
        return planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    def _filename_from_idx(self, idx: int) -> Tuple[Path, str]:
        fname = self.filename_list[idx]
        return fname, str(fname).replace('.xz', '-sym.txt')

    def read_points(self, idx: int) -> torch.Tensor:
        fname, _ = self._filename_from_idx(idx)
        with lzma.open(fname, 'rb') as fhandle:
            points = torch.tensor(np.loadtxt(fhandle))
        return points

    def read_planes(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        _, sym_fname = self._filename_from_idx(idx)
        return self._parse_sym_file(sym_fname)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> SymDatasetItem:
        fname, _ = self._filename_from_idx(idx)

        # ==========================================
        # NUEVO BLOQUE: DECIDIR DE DÓNDE LEER
        # ==========================================
        if self.load_to_ram:
            # Si están en RAM, los sacamos al instante de la lista
            points, planes = self.ram_cache[idx]
            planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries = planes
            # Es vital clonar los puntos para que las rotaciones no modifiquen la caché original
            points = points.clone() 
        else:
            # Si no hay RAM, lee lento del disco duro (comportamiento antiguo)
            points = self.read_points(idx)
            planar_symmetries = None
            axis_continue_symmetries = None
            axis_discrete_symmetries = None
            if self.has_ground_truth:
                planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries = self.read_planes(idx)

        idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries = self.transform(
            idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
        )

        transform_used = copy.deepcopy(self.transform)
        dataset_item = SymDatasetItem(
            fname.stem,
            idx, points.float(),
            planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries,
            transform_used
        )
        return dataset_item