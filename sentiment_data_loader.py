import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class SentimentDataset(Dataset):
    def __init__(self, X_path, y_path, seq_length):
        self.seq_length = seq_length
        with open(X_path, "rb") as f:
            self.X = pickle.load(f)
        with open(y_path, "rb") as f:
            self.y = pickle.load(f)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx][-self.seq_length:]  # Get last seq_length characters
        y = self.y[idx]

        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.int64)

        # Pad if needed
        if len(x_tensor) < self.seq_length:
            pad_size = self.seq_length - len(x_tensor)
            x_tensor = torch.cat([torch.zeros(pad_size, dtype=torch.int64), x_tensor])

        y_tensor = torch.tensor(y, dtype=torch.int64)

        return x_tensor, y_tensor
    
class DataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def get_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)  # Restart
            return next(self.iterator)