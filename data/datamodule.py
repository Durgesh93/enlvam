import os
import sys
import lightning as L
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from natsort import natsorted
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from dotenv import find_dotenv

envpath = find_dotenv()
sys.path.insert(0,os.path.dirname(envpath))

from data.dataset import Dataset
from data.iterabledataset import IterDataset
from typing import Dict, List, Any


class LitDatamodule(L.LightningDataModule):
    def __init__(self, cfg: Any):
        """
        Initialize the datamodule with configuration.
        """
        super().__init__()
        self.cfg = cfg
        self.src: Dict[str, Any] = self.__prepare_handlers()  # Mapping from dataloader ID to datasource

    def filter(self, type: str) -> List[Any]:
        """
        Filter dataloader configurations based on type (train/val/test).
        """
        if type == 'train':
            dl_cfg = list(filter(lambda cfg: cfg.id.split('/')[0] == 'train', self.cfg.dataloaders))
        else:
            dl_cfg = list(filter(lambda cfg: cfg.active and cfg.id.split('/')[0] == type, self.cfg.dataloaders))
        dl_cfg = natsorted(dl_cfg, key=lambda cfg: cfg.id)
        return dl_cfg

    def __prepare_handlers(self) -> Dict[str, Any]:
        """
        Instantiate and prepare data processing handlers for each dataloader.
        """
        data = {}
        for cfg in self.cfg.dataloaders:
            datasource = instantiate(cfg.datasource)
            transform = instantiate(cfg.transform)
            processing = instantiate(cfg.processing, datasource=datasource, transform=transform)
            data[cfg.id] = processing
        return data

    def _init_datasets(self) -> Dict[str, Any]:
        """
        Initialize datasets (either map-style or iterable) for each dataloader.
        """
        datasets = {}
        for cfg in self.cfg.dataloaders:
            datasource = self.src[cfg.id]
            if hasattr(datasource, '_generator'):
                dataset_type = 'iterable'
            else:
                dataset_type = 'map'

            if dataset_type == 'map':
                datasets[cfg.id] = Dataset(dataset_type, datasource)
            elif dataset_type == 'iterable':
                datasets[cfg.id] = IterDataset(dataset_type, datasource)
        return datasets

    def train_dataloader(self) -> CombinedLoader:
        """
        Return combined training dataloaders using max_size_cycle strategy.
        """
        self.datasets = self._init_datasets()
        train_cfgs = self.filter('train')
        train_dataloaders = []

        for cfg_dl in train_cfgs:
            dataset = self.datasets[cfg_dl.id]
            dl = DataLoader(
                dataset,
                batch_size=cfg_dl.batchsize,
                shuffle=True,
                pin_memory=cfg_dl.pin_memory,
                num_workers=cfg_dl.num_workers,
                drop_last=True
            )
            train_dataloaders.append(dl)

        return CombinedLoader(train_dataloaders, mode='max_size_cycle')

    def val_dataloader(self) -> List[DataLoader]:
        """
        Return a list of validation dataloaders.
        """
        self.datasets = self._init_datasets()
        val_cfgs = self.filter('val')
        val_dataloaders = []

        for cfg_dl in val_cfgs:
            dataset = self.datasets[cfg_dl.id]
            dl = DataLoader(
                dataset,
                batch_size=cfg_dl.batchsize,
                shuffle=False,
                pin_memory=cfg_dl.pin_memory,
                num_workers=cfg_dl.num_workers
            )
            val_dataloaders.append(dl)

        return val_dataloaders

    def test_dataloader(self) -> List[DataLoader]:
        """
        Return a list of test dataloaders.
        """
        self.datasets = self._init_datasets()
        test_cfgs = self.filter('test')
        test_dataloaders = []

        for cfg_dl in test_cfgs:
            dataset = self.datasets[cfg_dl.id]
            dl = DataLoader(
                dataset,
                batch_size=cfg_dl.batchsize,
                shuffle=False,
                pin_memory=cfg_dl.pin_memory,
                num_workers=cfg_dl.num_workers
            )
            test_dataloaders.append(dl)

        return test_dataloaders

    def predict_dataloader(self) -> List[DataLoader]:
        """
        Use test dataloaders for prediction.
        """
        return self.test_dataloader()
