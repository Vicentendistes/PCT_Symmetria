import copy
import lzma
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
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
            data_source_path: str = "path/to/dataset/split",
            transform: AbstractTransform = IdentityTransform(),
            has_ground_truth: bool = True,
            shape_excluded: Optional[List[str]] = None,
            perturbation_excluded: Optional[List[str]] = None,
            debug=False
    ):
        """
        Dataset used for a track of SHREC2023. It contains a set of 3D points
        and planes that represent reflective symmetries.
        :param data_source_path: Path to folder that contains the points and symmetries.
        :param transform: Transform applied to dataset item.
        """
        self.data_source_path = Path(data_source_path)
        self.transform = transform
        self.has_ground_truth = has_ground_truth
        self.debug = debug

        if self.debug:
            print(f'Searching xz-compressed point cloud files in {self.data_source_path}...')

        self.shape_excluded = [] if shape_excluded is None else shape_excluded
        self.perturbation_excluded = [] if perturbation_excluded is None else perturbation_excluded

        self.filename_list = list(self.data_source_path.rglob(f'*/*.xz'))

        self.filename_list = [
            filename for filename in self.filename_list
            if str(filename.name).split("-")[1] not in self.shape_excluded
               and str(filename.name).split("-")[2].replace(".xz", "") not in self.perturbation_excluded
        ]

        self.length = len(self.filename_list)
        if self.debug:
            print(
                f'{self.data_source_path.name}: found {self.length} files:\n{self.filename_list[:5]}\n{self.filename_list[-5:]}\n')

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
        if self.debug:
            print(f'Parsed file at: {filename}')
            formatted_print = lambda probably_tensor, text: print(f'\t No {text} found.') if probably_tensor is None \
                else print(f'\tGot {probably_tensor.shape[0]} {text}.')
            formatted_print(planar_symmetries, "Plane symmetries")
            formatted_print(axis_discrete_symmetries, "Discrete axis symmetries")
            formatted_print(axis_continue_symmetries, "Continue axis symmetries")

        return planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    def _filename_from_idx(self, idx: int) -> Tuple[Path, str]:
        if idx < 0 or idx >= len(self.filename_list):
            raise IndexError(f"Invalid index: {idx}, dataset size is: {len(self.filename_list)}")
        fname = self.filename_list[idx]
        if self.debug:
            print(f'Opening file: {fname.name}')
        return fname, str(fname).replace('.xz', '-sym.txt')

    def read_points(self, idx: int) -> torch.Tensor:
        """
        Reads the points with index idx.
        :param idx: Index of points to be read.
                    Not to be confused with the shape ID, this is now just the index in self.flist.
        :return: A tensor of shape N x 3 where N is the amount of points.
        """
        fname, _ = self._filename_from_idx(idx)

        with lzma.open(fname, 'rb') as fhandle:
            points = torch.tensor(np.loadtxt(fhandle))

        if self.debug:
            torch.set_printoptions(linewidth=200)
            torch.set_printoptions(precision=3)
            torch.set_printoptions(sci_mode=False)
            print(f'[{idx}]: {points.shape = }\n{points = }')

        return points

    def read_planes(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Read symmetry planes from file with its first line being the number of symmetry planes
        and the rest being the symmetry planes.
        :param idx: The idx of the syms to reads
        :return: A tensor of planes represented by their normals and points. N x 6 where
        N is the amount of planes and 6 because the first 3 elements
        are the normal and the last 3 are the point.
        """
        _, sym_fname = self._filename_from_idx(idx)
        return self._parse_sym_file(sym_fname)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> SymDatasetItem:
        fname, _ = self._filename_from_idx(idx)
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


if __name__ == "__main__":
    dataset = SymDataset("/data/sym-10k-xz-split-class-noparallel/train",
                         ComposeTransform(
                             [RandomSampler(sample_size=3),
                              UnitSphereNormalization()]
                         )
                         )
    xd = dataset[0]
    print(xd)
    print(xd.shape_type, xd.perturbation_type)
    print(xd.get_shape_type_classification_label())
