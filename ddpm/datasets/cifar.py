import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


__all__ = ["CifarDataset"]


class CifarDataset(Dataset):

    def __init__(self, dataset_dir, transform=None):
        dataset_paths = glob(os.path.join(dataset_dir, "*"))
        datas = []
        for data_path in dataset_paths:
            basename = os.path.basename(data_path)
            if basename.startswith("data"):
                data = unpickle(data_path)
                images = data[b"data"]
                datas.append(images)
        datas = np.stack(datas, axis=0)
        self.transform = transform
        self.images = datas.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)
