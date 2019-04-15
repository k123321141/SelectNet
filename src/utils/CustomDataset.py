from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(X) == len(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]
