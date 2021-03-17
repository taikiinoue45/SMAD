from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Tuple

from albumentations import Compose
from numpy import ndarray as NDArray
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset


class BaseRunner(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.transforms = {k: self._init_transforms(k) for k in self.cfg.transforms.keys()}
        self.datasets = {k: self._init_datasets(k) for k in self.cfg.datasets.keys()}
        self.dataloaders = {k: self._init_dataloaders(k) for k in self.cfg.dataloaders.keys()}

    def _init_transforms(self, key: str) -> Compose:

        transforms = []
        for cfg in self.cfg.transforms[key]:
            attr = self._get_attr(cfg.name)
            transforms.append(attr(**cfg.get("args", {})))
        return Compose(transforms)

    def _init_datasets(self, key: str) -> Dataset:

        cfg = self.cfg.datasets[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), transforms=self.transforms[key])

    def _init_dataloaders(self, key: str) -> DataLoader:

        cfg = self.cfg.dataloaders[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), dataset=self.datasets[key])

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)

    def run(self) -> None:

        D, X = self._train()
        self._test(D, X)

    @abstractmethod
    def _train(self) -> Tuple[NDArray, NDArray]:

        raise NotImplementedError()

    @abstractmethod
    def _test(self, D: NDArray, X: NDArray) -> None:

        raise NotImplementedError()
