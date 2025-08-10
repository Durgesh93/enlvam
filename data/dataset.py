from torch.utils import data

class Dataset(data.Dataset):
    
    def __init__(self, type, datasource):
        # Store the type of dataset (e.g., 'train', 'val', 'test')
        self.datasource = datasource
        self.type = type

    def __len__(self):
        # Return the number of samples based on the datasource's split
        return len(self.datasource.split)

    def __getitem__(self, idx):
        # Initialize empty input and label dictionaries
        X = {}
        y = {'idx': idx}

        # Convert raw data to tensors using the datasource's method
        X, y = self.datasource.to_tensor(X, y)

        # Return a dictionary containing input and label tensors
        return {'X': X, 'y': y}
