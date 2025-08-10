import pandas as pd
import fsspec
import tempfile
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import MLFlowLogger
from typing import Any
from PIL import Image
import numpy as np

class ExtendedMLFlowLogger(MLFlowLogger):
    """
    A custom extension of MLFlowLogger that supports offline logging and 
    logging pandas DataFrames as CSV artifacts using fsspec and temporary files.
    """

    def __init__(self, *args: Any, mode: str = 'offline', **kwargs: Any) -> None:
        """
        Initializes the ExtendedMLFlowLogger.

        Parameters:
        - mode (str): Determines the logging mode. If 'offline', logs are stored locally.
        - *args, **kwargs: Additional arguments passed to the MLFlowLogger constructor.
        """
        self.mode: str = mode

        # If operating in offline mode, create a temporary directory for MLflow tracking
        if self.mode == 'offline':
            self.temp_dir = tempfile.TemporaryDirectory()
            # Set MLflow tracking URI to the temporary directory
            kwargs['tracking_uri'] = f"{self.temp_dir.name}"

        # Initialize the base MLFlowLogger with updated arguments
        super().__init__(*args, **kwargs)

    def log_table(self, k: str, dataframe: pd.DataFrame) -> None:
        """
        Logs a pandas DataFrame as a CSV artifact to MLflow.

        Parameters:
        - k (str): A key or name used as the artifact path in MLflow.
        - dataframe (pd.DataFrame): The DataFrame to be logged.
        """
        # Define the path in the in-memory filesystem using fsspec
        mem_path: str = f"memory://split/{k}/table.csv"

        # Write the DataFrame to the in-memory filesystem as a CSV
        with fsspec.open(mem_path, mode='w') as f:
            dataframe.to_csv(f, index=False)

        # Read the CSV from memory and write it to a temporary file
        with fsspec.open(mem_path, mode='rb') as mem_file:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
                # Copy the contents from memory to the temp file
                tmp_file.write(mem_file.read())
                tmp_file.flush()

                # Log the temporary CSV file as an artifact in MLflow
                self.experiment.log_artifact(
                    run_id=self.run_id,
                    local_path=tmp_file.name,
                    artifact_path=f"{k}"
                )
    
    def log_image(self, k: str, image_array: np.ndarray,current_epoch:int,current_step:int) -> None:
        """
        Logs a NumPy image array as an image artifact to MLflow.

        Parameters:
        - k (str): A key or name used as the artifact path in MLflow.
        - image_array (np.ndarray): The image data as a NumPy array (H x W x C or H x W).
        """
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image_array.astype(np.uint8))

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=f"_epoch_{current_epoch}_step_{current_step}.png", delete=False) as tmp_file:
            image.save(tmp_file.name)

            # Log the image file as an artifact in MLflow
            self.experiment.log_artifact(
                run_id=self.run_id,
                local_path=tmp_file.name,
                artifact_path=k
            )


