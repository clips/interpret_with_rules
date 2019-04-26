from torch.utils.data import Dataset
import torch

class SparseDataset(Dataset):
    '''
    Dataset object for scipy sparse matrices
    '''
    def __init__(self, feats_sparse, labels = None, transforms = None):
        self.x = feats_sparse
        if labels is not None:
            self.y = torch.from_numpy(labels)
        self.transforms = transforms

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x.getrow(idx).toarray().reshape(-1,)).float()
        if self.transforms:
            x = self.transforms(x)

        try:
            y = self.y[idx]
            return (x, y)
        except:
            return x

    def __len__(self):
        return self.x.shape[0]

class CustomDataset(Dataset):
    '''
    Dataset object for numpy dense matrices
    '''
    def __init__(self, feats, labels = None, transforms=None):
        self.x = torch.from_numpy(feats)
        if labels is not None:
            self.y = torch.from_numpy(labels)
        self.transforms = transforms

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transforms:
            x = self.transforms(x)

        try:
            y = self.y[idx]
            return (x, y)
        except:
            return x

    def __len__(self):
        return self.x.shape[0]