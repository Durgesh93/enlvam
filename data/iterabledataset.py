from typing import Any, Dict, Iterator
from torch.utils import data

# A simple sentinel class used to signal the end of the data stream
class Sentinel:
    def __init__(self) -> None:
        pass

# Custom iterable dataset for streaming data from a datasource
class IterDataset(data.IterableDataset):
    def __init__(self, type: str, datasource: Any) -> None:
        """
        Args:
            type (str): A string indicating the type or mode of the dataset.
            datasource (Any): An object that provides data via `split` and `to_tensor` methods.
        """
        super().__init__()
        self.datasource: Any = datasource
        self.type: str = type
        self.sentinel: Sentinel = Sentinel()  # Sentinel object to detect end of stream

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterator that yields data samples in the form of dictionaries with 'X' and 'y' keys.
        """
        idx: int = 0  # Initialize index counter

        while True:
            # Attempt to get the next item from the datasource
            item = next(self.datasource.split, self.sentinel)

            # If the item is the sentinel, we've reached the end of the stream
            if isinstance(item, Sentinel):
                break

            # Convert the item to a dictionary
            row_info: Dict[str, Any] = item.to_dict()

            # Prepare input (X) and target (y) dictionaries
            X: Dict[str, Any] = {}
            y: Dict[str, Any] = row_info
            y['idx'] = idx  # Add index to the target dictionary

            idx += 1  # Increment index

            # Convert raw data to tensors using the datasource's method
            X, y = self.datasource.to_tensor(X, y)

            # Yield the sample as a dictionary
            yield {'X': X, 'y': y}
