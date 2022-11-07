from torch.utils.data.dataset import Dataset
import numpy as np

class SingleStockDataset(Dataset):
    def __init__(self, x_data, y_label):
        x_data = np.expand_dims(x_data, 2)
        self.data = x_data.astype(np.float32)
        self.label = y_label.astype(np.float32)

    def __getitem__(self, item):

        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)

