from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset


class BatchedFeaturesDataset(Dataset):

    def __init__(
            self, 
            features_dict:Dict[int,torch.Tensor], 
            targets_df:pd.DataFrame, 
            target_cols:str,
            id_col:str='image_ids',
            transform=None
        ):
        self.features_dict = features_dict
        self.targets_df = targets_df
        self.target_cols = target_cols
        self.id_col = id_col
        self.transform = transform

    def __len__(self):
        return len(self.targets_df)
    
    def __getitem__(self, idx):
        row = self.targets_df.iloc[idx]
        features = torch.stack([self.features_dict[i] for i in row[self.id_col]])
        if self.transform is not None:
            features = self.transform(features)
        target_dict = {col: torch.tensor(row[col], dtype=torch.float32) for col in self.target_cols}
        return features, target_dict
    
    def collate_fn(self, batch):
        features, targets = zip(*batch)
        features = torch.stack(features)
        targets = {
            k: torch.stack([t[k] for t in targets]).view(-1, 1)
            for k in targets[0].keys()}
        return features, targets
    
