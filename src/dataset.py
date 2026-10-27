import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

MEAN, STD = 0.35, 0.25  # approximate pixel statistics of the synthetic data


class SEMDataset(Dataset):
    def __init__(self, X, y, train=False):
        self.X, self.y, self.train = X, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.X[i].astype(np.float32) / 255.0
        if self.train:  # label-preserving augmentations
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1, :]
            img = np.rot90(img, np.random.randint(4))
            img = img * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05)
        img = (np.ascontiguousarray(img) - MEAN) / STD
        return torch.from_numpy(img).unsqueeze(0).float(), int(self.y[i])


def load_splits(path, seed=42):
    d = np.load(path)
    X, y = d["X"], d["y"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    classes = [str(c) for c in d["classes"]]
    return (SEMDataset(X_tr, y_tr, True), SEMDataset(X_va, y_va), SEMDataset(X_te, y_te)), classes
