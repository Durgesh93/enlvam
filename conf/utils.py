import torch
import os
from types import SimpleNamespace
from typing import Union, Any, Dict, List
from omegaconf import OmegaConf, DictConfig, ListConfig

from .fake import experiment_name


class MisconfigurationException(Exception):
    """Custom exception for configuration errors."""
    def __init__(self, message: str = "There is a misconfiguration in your setup.") -> None:
        self.message: str = message
        super().__init__(self.message)


def acc() -> str:
    """
    Determines the available device for computation.

    Returns:
        str: 'gpu' if CUDA is available, otherwise 'cpu'.
    """
    return 'gpu' if torch.cuda.is_available() else 'cpu'


def jobname() -> str:
    """
    Retrieves the job name from environment variables or generates a default name.

    Returns:
        str: Job name.
    """
    return os.environ.get('JOB_NAME', experiment_name())


def init_resolvers() -> None:
    """
    Registers custom resolvers with OmegaConf if they are not already registered.
    """
    if OmegaConf._get_resolver("oc.ifelse") is None:
        OmegaConf.register_new_resolver(
            "oc.ifelse",
            lambda curr, tvalue, topt, fopt: topt if curr == tvalue else fopt
        )

    if OmegaConf._get_resolver("oc.acc") is None:
        OmegaConf.register_new_resolver("oc.acc", acc)

    if OmegaConf._get_resolver("oc.jobname") is None:
        OmegaConf.register_new_resolver("oc.jobname", jobname)

    if OmegaConf._get_resolver("oc.eval") is None:
        OmegaConf.register_new_resolver("oc.eval", lambda expression: eval(expression))


def get_namespace(dc: Union[DictConfig, ListConfig], resolve: bool) -> SimpleNamespace:
    """
    Converts a DictConfig or ListConfig into a nested SimpleNamespace.

    Args:
        dc (DictConfig or ListConfig): The configuration object.
        resolve (bool): Whether to resolve interpolations.

    Returns:
        SimpleNamespace: A namespace representation of the configuration.
    """
    container: Union[Dict[str, Any], List[Any]] = OmegaConf.to_container(dc, resolve=resolve)

    def ns(d: Any) -> Any:
        if isinstance(d, dict):
            return SimpleNamespace(**{key: ns(value) for key, value in d.items()})
        elif isinstance(d, list):
            return SimpleNamespace(**{f'index_{idx}': ns(item) for idx, item in enumerate(d)})
        else:
            return d

    return ns(container)


def flatten_dict(config: Union[Dict[str, Any], DictConfig, List[Any], ListConfig], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary or OmegaConf config into a single-level dictionary.

    Args:
        config (dict or DictConfig): The configuration to flatten.
        parent_key (str): The base key to prefix.
        sep (str): Separator between keys.

    Returns:
        dict: A flattened dictionary.
    """
    items: List[tuple[str, Any]] = []

    if isinstance(config, (dict, DictConfig)):
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.extend(flatten_dict(v, new_key, sep=sep).items())
    elif isinstance(config, (list, ListConfig)):
        for i, v in enumerate(config):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, config))

    return dict(items)
