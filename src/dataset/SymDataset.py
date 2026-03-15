import copy
from pathlib import Path
from typing import Optional, List, Tuple
import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.dataset.SymDatasetItem import SymDatasetItem
from src.dataset.transforms.AbstractTransform import AbstractTransform
from src.dataset.transforms.ComposeTransform import ComposeTransform
from src.dataset.transforms.IdentityTransform import IdentityTransform
from src.dataset.transforms.RandomSampler import RandomSampler
from src.dataset.transforms.UnitSphereNormalization import UnitSphereNormalization

class SymDataset(Dataset):      
    def __init__(
            self,
            data_source_path: str, # Ahora esto recibirá la ruta al archivo .h5 (ej: train.h5)
            transform: AbstractTransform = IdentityTransform(),
            has_ground_truth: bool = True,
            shape_excluded: Optional[List[str]] = None,
            perturbation_excluded: Optional[List[str]] = None,
            debug=False
    ):
        self.h5_path = Path(data_source_path)
        self.transform = transform
        self.has_ground_truth = has_ground_truth
        self.debug = debug
        self.shape_excluded = [] if shape_excluded is None else shape_excluded
        self.perturbation_excluded = [] if perturbation_excluded is None else perturbation_excluded
        
        # Puntero al archivo (se abre en el primer __getitem__ por los workers)
        self.h5_file = None

        if self.debug:
            print(f'Leyendo índice de HDF5 en {self.h5_path}...')

        # Abrimos solo para sacar los nombres de los modelos (keys)
        with h5py.File(self.h5_path, 'r') as f:
            all_keys = list(f.keys())

        # Filtrar shapes y perturbaciones exactamente como lo hacías antes
        self.keys = []
        for key in all_keys:
            parts = key.split("-")
            if len(parts) >= 3:
                if parts[1] not in self.shape_excluded and parts[2] not in self.perturbation_excluded:
                    self.keys.append(key)
            else:
                self.keys.append(key)

        self.length = len(self.keys)
        if self.debug:
            print(f'{self.h5_path.name}: encontrados {self.length} modelos válidos.')

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> SymDatasetItem:
        # TRUCO MULTIPROCESSING: Abrir el archivo dentro del worker
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        shape_id = self.keys[idx]
        group = self.h5_file[shape_id]

        # 1. Leer Puntos
        points = torch.tensor(group['points'][:])

        planar_symmetries = None
        axis_continue_symmetries = None
        axis_discrete_symmetries = None

        # 2. Leer Simetrías
        if self.has_ground_truth:
            p_sym = group['planar_symmetries'][:]
            ac_sym = group['axis_continue_symmetries'][:]
            ad_sym = group['axis_discrete_symmetries'][:]

            if len(p_sym) > 0: planar_symmetries = torch.tensor(p_sym)
            if len(ac_sym) > 0: axis_continue_symmetries = torch.tensor(ac_sym)
            if len(ad_sym) > 0: axis_discrete_symmetries = torch.tensor(ad_sym)

        # 3. Aplicar Transforms (¡Tu lógica intacta!)
        idx_trans, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries = self.transform(
            idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
        )

        transform_used = copy.deepcopy(self.transform)

        # 4. Retornar el item final
        dataset_item = SymDatasetItem(
            shape_id,
            idx_trans, points.float(),
            planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries,
            transform_used
        )

        return dataset_item