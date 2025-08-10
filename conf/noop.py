import torch
import numpy as np
import pandas as pd
from typing import Any, Dict,List,Any, Tuple, Optional

# Dummy class for conf.noop.Splitter
class Splitter:
    def __init__(self, metadata:pd.DataFrame, split_name: str = 'all', filter_rule: str = 'default', labeled_fraction: float = 1.0) -> None:
        self.metadata: pd.DataFrame = metadata
        self.split_name: str = split_name
        self.filter_rule: str = filter_rule
        self.labeled_fraction: float = labeled_fraction

    def get_split(self) -> pd.DataFrame:
        return self.metadata

# Dummy class for conf.noop.DataSource
class DataSource:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg: Dict[str, Any] = cfg
        self.metadata   = self.get_metadata()
        self.split      = Splitter(
                            self.metadata,
                            **self.cfg.splitter
                        ).get_split()
        
    def get_metadata(self) -> pd.DataFrame:
        data = {
            "id": [1, 2, 3, 4, 5],
            "name": ["Item A", "Item B", "Item C", "Item D", "Item E"],
            "type": ["Type 1", "Type 2", "Type 1", "Type 3", "Type 2"],
            "value": [10.5, 23.0, 7.8, 15.2, 9.1]
        }
        return pd.DataFrame(data)

    def __getitem__(self,dataid):
        raise  NotImplementedError('get item is not implemented for this object')

    def __len__(self):
        return len(self.split)


# Dummy class for conf.noop.Processing
class Processing:
    def __init__(self, datasource: Any, transform: Any, cfg: Optional[dict] = None) -> None:
        """
        Initialize the Processing class.

        Args:
            datasource (Any): The data source object.
            transform (Any): The transformation logic or object.
            cfg (Optional[dict]): Configuration dictionary (can be None).
        """
        self.cfg = cfg
        self.datasource = datasource
        self.transform = transform
        self.split     = self.create_split(self.datasource.split)

    def create_split(self, split: Any) -> Any:
        split = pd.concat([split, split], ignore_index=True)  
        return split
    
    def pre_process(self, X: Any, y: Any) -> Tuple[Any, Any]:
        return X, y

    def post_process(self, X: Any, y: Any) -> Tuple[Any, Any]:
        return X, y

    def to_tensor(self, X, y, batchmode=False):
        X_tensors = {}
        y_tensors = {}
        if not batchmode:
            for k, v in X.items():
                X[k] = np.array([v])
            for k, v in y.items():
                y[k] = np.array([v])

        for k, v in X.items():
            if isinstance(v,list):
                X_tensors[k] = torch.from_numpy(np.array(v))
            elif isinstance(v,np.ndarray):
                X_tensors[k] = torch.from_numpy(v)

        for k, v in y.items():
            if isinstance(v,list):
                y_tensors[k] = torch.from_numpy(np.array(v))
            elif isinstance(v,np.ndarray):
                y_tensors[k] = torch.from_numpy(v)

        if not batchmode:
            for k, v in X_tensors.items():
                X_tensors[k] = v[0]
            for k, v in y_tensors.items():
                y_tensors[k] = v[0]
                
        return X_tensors,y_tensors

# Dummy class for conf.noop.Transform
class Transform:
    def __init__(self, cfg: dict) -> None:
        """
        Initialize the Transform class.

        Args:
            cfg (dict): Configuration dictionary (expected to be empty for noop).
        """
        self.cfg: dict = cfg

    def apply(self, data: any) -> any:
        """
        Dummy transform method that returns the input data unchanged.

        Args:
            data (any): Input data to transform.

        Returns:
            any: Unchanged input data.
        """
        print("Transform: no transformation applied.")
        return data


# Dummy class for conf.noop.Logger
class Logger:
    def __init__(self, cfg: dict) -> None:
        """
        Initialize the Logger class.

        Args:
            cfg (dict): Configuration dictionary (expected to be empty for noop).
        """
        self.cfg: dict = cfg

    def log(self, message: str) -> None:
        """
        Dummy log method that prints the message to the console.

        Args:
            message (str): The message to log.
        """
        print(f"NoopLogger: {message}")


# Dummy class for conf.noop.Pbar
class Pbar:
    def __init__(self, cfg: dict) -> None:
        """
        Initialize the Pbar class.

        Args:
            cfg (dict): Configuration dictionary (expected to be empty for noop).
        """
        self.cfg: dict = cfg

    def update(self, progress: int, total: int) -> None:
        """
        Dummy progress bar update method.

        Args:
            progress (int): Current progress value.
            total (int): Total value for completion.
        """
        print(f"NoopPbar: progress {progress}/{total}")

    def close(self) -> None:
        """
        Dummy method to close the progress bar.
        """
        print("NoopPbar: closed.")


# Dummy class for conf.noop.Summary
class Summary:
    def __init__(self, max_depth: int = 4) -> None:
        """
        Initialize the Summary class.

        Args:
            max_depth (int): Maximum depth for summary traversal.
        """
        self.max_depth: int = max_depth

    def summarize(self, data: any) -> str:
        """
        Dummy summarize method that returns a placeholder summary.

        Args:
            data (any): Input data to summarize.

        Returns:
            str: A placeholder summary string.
        """
        print(f"NoopSummary: summarizing with max_depth={self.max_depth}")
        return "Summary not implemented (noop)."


# Dummy class for conf.noop.Trainer
class Trainer:
    def __init__(self, logger: str = "noop", callbacks: list = None, cfg: dict = None) -> None:
        """
        Initialize the Trainer class.

        Args:
            logger (str): Logger identifier (default is 'noop').
            callbacks (list): List of callback objects (default is empty list).
            cfg (dict): Configuration dictionary (default is empty dict).
        """
        self.logger: str = logger
        self.callbacks: list = callbacks if callbacks is not None else []
        self.cfg: dict = cfg if cfg is not None else {}

    def train(self, data: any) -> None:
        """
        Dummy train method that simulates training.

        Args:
            data (any): Input training data.
        """
        print(f"NoopTrainer: training started with logger='{self.logger}', "
              f"{len(self.callbacks)} callbacks, and cfg={self.cfg}")
        print("NoopTrainer: training completed.")


# Dummy class for conf.noop.Loss
class Loss:
    def __init__(self, cfg: dict) -> None:
        """
        Initialize the Loss class.

        Args:
            cfg (dict): Configuration dictionary (expected to be empty for noop).
        """
        self.cfg: dict = cfg

    def compute(self, predictions: any, targets: any) -> float:
        """
        Dummy loss computation method that returns a constant value.

        Args:
            predictions (any): Model predictions.
            targets (any): Ground truth targets.

        Returns:
            float: A placeholder loss value.
        """
        print("NoopLoss: returning constant loss value.")
        return 0.0


# Dummy class for conf.noop.Metrics


class Metrics:
    def __init__(self, cfg: dict) -> None:
        """
        Initialize the Metrics class.

        Args:
            cfg (dict): Configuration dictionary containing metric parameters.
        """
        self.params: List[Any] = cfg.get("params", [])

    def compute(self, predictions: Any, targets: Any) -> dict:
        """
        Dummy compute method that returns a placeholder metric result.

        Args:
            predictions (Any): Model predictions.
            targets (Any): Ground truth targets.

        Returns:
            dict: A dictionary with dummy metric values.
        """
        print(f"NoopMetrics: computing metrics with params={self.params}")
        return {"accuracy": 1.0, "loss": 0.0}

# Dummy class for conf.noop.Model
class Model:
    def __init__(self, cfg: dict) -> None:
        """
        Initialize the Model class.

        Args:
            cfg (dict): Configuration dictionary (expected to be empty for noop).
        """
        self.cfg: dict = cfg

    def forward(self, input_data: Any) -> Any:
        """
        Dummy forward method that returns the input data unchanged.

        Args:
            input_data (Any): Input data to the model.

        Returns:
            Any: Unchanged input data.
        """
        print("NoopModel: forwarding input without modification.")
        return input_data

# Dummy class for conf.noop.Optimizer
class Optimizer:
    def __init__(self, lr: float = 0.1, weight_decay: float = 0.1, params: List[Any] = None) -> None:
        """
        Initialize the Optimizer class.

        Args:
            lr (float): Learning rate.
            weight_decay (float): Weight decay factor.
            params (List[Any]): List of parameters to optimize.
        """
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.params: List[Any] = params if params is not None else []

    def step(self) -> None:
        """
        Dummy step method to simulate an optimization step.
        """
        print(f"NoopOptimizer: performing step with lr={self.lr}, weight_decay={self.weight_decay}")
        print(f"NoopOptimizer: optimizing {len(self.params)} parameters.")

    def zero_grad(self) -> None:
        """
        Dummy method to simulate gradient reset.
        """
        print("NoopOptimizer: gradients reset.")


# Dummy class for conf.noop.Scheduler
class Scheduler:
    def __init__(self, optimizer: Any, gamma: float = 0.1) -> None:
        """
        Initialize the Scheduler class.

        Args:
            optimizer (Any): Optimizer instance to be scheduled.
            gamma (float): Decay factor for learning rate.
        """
        self.optimizer: Any = optimizer
        self.gamma: float = gamma

    def step(self) -> None:
        """
        Dummy step method to simulate learning rate scheduling.
        """
        print(f"NoopScheduler: applying decay with gamma={self.gamma} to optimizer.")
