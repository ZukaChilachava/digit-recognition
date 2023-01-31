import torch
import numpy as np
from torch.utils.data import Dataset


class LetterDataset(Dataset):
    def __init__(self, amount, start, path: str, delimiter: str = ","):
        end_range = start + amount

        y = np.genfromtxt(fname=path, delimiter=delimiter, dtype=None, encoding="UTF-8", usecols=0)[start:end_range]
        x = np.loadtxt(fname=path, delimiter=delimiter, dtype=np.float32, usecols=range(1, 17))[start:end_range]

        self.x = torch.as_tensor(x)
        # convert output letters to their indexes 0-26
        self.y = torch.as_tensor([ord(curr_y) - ord('A') for curr_y in np.asarray(y)], dtype=torch.float32)
        self.n_samples = self.y.shape[0]

    def __getitem__(self, item):
        return self.y[item], self.x[item]

    def __len__(self):
        return self.n_samples
