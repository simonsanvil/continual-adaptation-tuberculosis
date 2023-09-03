from typing import List, Dict
import numpy as np

import torch
import torchvision
import torchvision.transforms as T

from .dataset import BaciliDataset

class BaciliDatasetPytorch(torch.utils.data.Dataset):
    
    def __init__(self, dataset:List[Dict], train:bool=False):
        self.dataset = dataset
        self.transforms = self.get_transform(train)

    def get_transform(self, train:bool=True):
        """
        Define the transformation of the images
        """
        # TODO: add transforms
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get the image and annotations
        """
        item = self.dataset[idx]
        img = item["image"]
        target = {
            "boxes": torch.tensor(item["boxes"], dtype=torch.float32),
            "labels": torch.tensor(item["labels"], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": self._calculare_areas(item["boxes"]),
        }
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    
    def get_height_and_width(self):
        return BaciliDataset.IMG_HEIGHT, BaciliDataset.IMG_WIDTH

    def _calculare_areas(self, boxes:np.ndarray):
        """
        Calculate the areas of the boxes
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        