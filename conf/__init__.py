import os
from typing import Optional, List, Union
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig

from .utils import (
                    init_resolvers,
                    flatten_dict,
                    MisconfigurationException
                )


def build_config(config_path: Optional[str] = None, cli_overrides: List[str] = []) -> DictConfig:
    """
    Build and return a merged experiment configuration using Hydra and OmegaConf.
    Supports CLI overrides for both options and config.

    Args:
        config_path (Optional[str]): Path to the Hydra config file.
        cli_overrides (List[str]): List of CLI-style overrides (e.g., ['c.lr=0.01']).

    Returns:
        DictConfig: The final merged configuration under the 'c' namespace.
    """
    # Initialize custom resolvers for OmegaConf (defined in utils.py)
    init_resolvers()

    # Initialize Hydra and compose the base configuration
    with initialize(version_base=None, config_path='.', job_name='echoAI'):
        
        options = compose(config_name='options')
        
        # find global cli overrides which starts with o.
        options_overrides = [
            cfg.replace('o.', '') for cfg in cli_overrides if cfg.startswith('o.')
        ]

        options_overrides = OmegaConf.from_dotlist(options_overrides)

        # merging the cli overrides to current global options.
        options = OmegaConf.merge(options, options_overrides)

        
        # if no config_path is provided return the options
        if config_path is None:
            OmegaConf.resolve(options.paths)
            return options.paths
        
    # Load experiment-specific config
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)
    

    with initialize(version_base=None, config_path=config_dir, job_name='echoAI'):
        config = compose(config_name=config_name)
        # find experiment related cli overrides that start with c.
        e_overrides = [
            cfg.replace('c.', '') for cfg in cli_overrides if cfg.startswith('c.')
        ]

        e_overrides = OmegaConf.from_dotlist(e_overrides)
        config = OmegaConf.merge(config, e_overrides)

    # Create an empty experiment config under the 'c' namespace
    experiment: DictConfig = OmegaConf.from_dotlist([])
    if 'hparams' in config:
        for k, v in config.hparams.items():
            experiment[k] = v

    experiment.hparams   = flatten_dict(config)
    experiment.echotools = options.modules.echotools
    experiment.paths     = options.paths

    metrics_dict = {}

    # Configure datamodule
    if 'datamodule' in config:
        experiment.datamodule = config.datamodule.copy()

        if 'dataloaders' in experiment.datamodule.cfg:
            for idx, cfg in enumerate(experiment.datamodule.cfg.dataloaders):
                
                if 'id' not in cfg:
                    raise MisconfigurationException(f"dataloader {idx} must have an 'id'")
                if 'datasource' not in cfg:
                    raise MisconfigurationException(f"dataloader {idx} must have a 'datasource'")

                cfg.setdefault('active', True)
                cfg.setdefault('split_name', 'all')
                cfg.setdefault('filter_rule', 'no_filter')
                cfg.setdefault('labeled_fraction', 1.0)
                cfg.setdefault('processing', 'noop')
                cfg.setdefault('transform', 'noop')
                cfg.setdefault('batchsize', 1)

                
                cfg.pin_memory = options.pin_memory
                cfg.num_workers = options.num_workers_per_dataloader

                cfg.datasource = options.data.datasources[cfg.datasource].copy()
                cfg.datasource.cfg.splitter.split_name = cfg.split_name
                cfg.datasource.cfg.splitter.filter_rule = cfg.filter_rule
                cfg.datasource.cfg.splitter.labeled_fraction = cfg.labeled_fraction
                cfg.datasource.cfg.echotools = experiment.echotools.copy()

                cfg.processing = options.data.processing[cfg.processing].copy()
                cfg.processing.cfg.echotools = experiment.echotools.copy()
                cfg.transform = options.data.transforms[cfg.transform].copy()

                if 'metrics' in cfg:
                    metrics_dict[cfg.id] = options.metrics[cfg.metrics].copy()
                    del cfg['metrics']

    # Configure litmodel
    if 'litmodel' in config:
        experiment.litmodel = config.litmodel.copy()

        if 'model' not in experiment.litmodel.cfg:
            raise MisconfigurationException('Torch model required for litmodel configuration')
        if 'optimizer' not in experiment.litmodel.cfg:
            raise MisconfigurationException('Optimizer required for litmodel configuration')
        if 'loss' not in experiment.litmodel.cfg:
            raise MisconfigurationException('Loss required for litmodel configuration')

        experiment.litmodel.cfg.hparams = experiment.hparams
        experiment.litmodel.cfg.paths = experiment.paths
        experiment.litmodel.cfg.metrics = OmegaConf.create(metrics_dict)
        experiment.litmodel.cfg.echotools = experiment.echotools.copy()
        experiment.litmodel.cfg.model = options.models[experiment.litmodel.cfg.model].copy()
        experiment.litmodel.cfg.optimizer = options.optimizers[experiment.litmodel.cfg.optimizer].copy()
        experiment.litmodel.cfg.scheduler = options.schedulers[experiment.litmodel.cfg.scheduler].copy()
        experiment.litmodel.cfg.loss = options.losses[experiment.litmodel.cfg.loss].copy()

    # Configure trainer
    if 'trainer' in config:
        if 'name' not in config.trainer:
            raise MisconfigurationException('Trainer name is required')

        experiment.trainer = options.lightning.trainers[config.trainer.name].copy()
        if 'logger' in config.trainer and config.trainer.logger:
            logger_cfg = options.lightning.loggers[config.trainer.logger]
            experiment.trainer.logger = logger_cfg

        trainer_callbacks = []

        if 'checkpointer' in config.trainer and config.trainer.checkpointer.enable:
            checkpointer = options.lightning.checkpointer.copy()
            if 'metric' in config.trainer.checkpointer and config.trainer.checkpointer.metric:
                checkpointer.monitor = config.trainer.checkpointer.metric
                checkpointer.mode = config.trainer.checkpointer.mode
            trainer_callbacks.append(checkpointer)

        if 'pbar' in config.trainer and config.trainer.pbar:
            pbar = options.lightning.pbars[config.trainer.pbar]
            summary = options.lightning.summary[config.trainer.pbar]
            trainer_callbacks.extend([pbar, summary])

        experiment.trainer.callbacks = trainer_callbacks

    # Finalize and return experiment config
    experiment = OmegaConf.create({'c': experiment})

    # Resolve interpolations and return the 'c' section
    OmegaConf.resolve(experiment)
    return experiment.c
