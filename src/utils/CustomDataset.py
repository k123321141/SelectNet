from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, Y, x_transforms=None, y_transforms=None):
        self.X = X
        self.Y = Y
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms

        assert len(X) == len(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx, :]
        if self.x_transforms:
            x = self.x_transforms(x)
        if self.y_transforms:
            y = self.y_transforms(y)
        return x, y
