import os, glob, sys, json, contextlib
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

import pandas as pd
import torch, wandb
from torch import nn, optim
from ml_collections.config_dict import ConfigDict
from sklearn.metrics import mean_squared_error, r2_score

from bacili_detection.active_learning.strategies.loss_prediction import DETRLossPredictor
from bacili_detection.utils.transforms import RandomSelect, AvgFeaturesTransform, RandomSampleFeaturesTransform
from bacili_detection.utils.datasets import BatchedFeaturesDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader

CONFIG_MODULE =  "bacili_detection.active_learning.experiments.configs"
DEFAULT_CONFIG = "loss_prediction_detr"

def main(config:ConfigDict):
    """
    Train the loss prediction model
    """
    # 1. build the datasets
    train_ds, test_ds = build_datasets(config)
    # 2. build the model
    model = build_model(config)
    # 3. train the model
    train(model, train_ds, test_ds, config)

def train(model:DETRLossPredictor, train_ds:BatchedFeaturesDataset, test_ds:BatchedFeaturesDataset, config:ConfigDict):
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    lr = config.lr
    device = config.device
    model_save_dir = config.model_save_dir
    eval_every = config.eval_every if config.eval_every is not None else num_epochs
    stop_patience = config.stop_patience
    lrs_patience = config.lrs_patience

    # dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=test_ds.collate_fn)

    # make criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # setup
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lrs_patience, verbose=True, factor=0.1)
    best_loss = float('inf')
    patience_counter = 0
    if config.log_wandb:
        wandb_config={
            "model": "loss-prediction:detr",
            "batch_size": batch_size, "num_epochs": num_epochs,
            "num_train": len(train_ds), "num_test": len(test_ds),
            "input_size": config.feature_size,
            "optimizer": optimizer.__class__.__name__,
            "lr": optimizer.param_groups[0]['lr'],
            "hidden_size": model.hidden_size,
        }
        wandb_init(config.wandb_project, model, **wandb_config)

    # training loop
    config['samples_seen'] = 0
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, config=config)
        train_loss /= len(train_dl)
        if epoch % eval_every != 0:
            continue
        metrics_dict = evaluate(model, test_dl, criterion, device=device)
        test_loss = metrics_dict['loss']
        print(f'Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}')
        print(f"test - R2 score: {test_loss['r2']:.2f} - RMSE: {test_loss['rmse']:.2f}")
        lr_scheduler.step(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        if patience_counter >= stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
def train_one_epoch(model:DETRLossPredictor, train_dl:DataLoader, criterion, optimizer, config:ConfigDict):
    train_loss = 0
    
    model.train()
    for ind, (X, target) in enumerate(train_dl):
        X = X.to(config.device)
        y = target[config.target_loss].to(config.device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        config['samples_seen'] = config['samples_seen'] + len(X)
        if config.wandb_log:
            if ind % 10 == 0:
                wandb.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']}, step=config['samples_seen'])

    train_loss /= len(train_dl)
    return train_loss

@torch.no_grad()
def evaluate(model:DETRLossPredictor, test_dl:DataLoader, criterion, config:ConfigDict):
    test_loss = 0
    model.eval()
    
    ys, y_preds = [], []
    for ind, (X, target) in enumerate(test_dl):
        X = X.to(config.device)
        y = target[config.target_loss].to(config.device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        test_loss += loss.item()
        ys.append(y.cpu().numpy())
        y_preds.append(y_pred.cpu().numpy())
        if config.wandb_log:
            if ind % 10 == 0:
                wandb.log({"test_loss": loss.item()}, step=config['samples_seen'])
        
    test_loss /= len(test_dl)
    ys = np.concatenate(ys)
    y_preds = np.concatenate(y_preds)
    r2 = r2_score(np.concatenate(ys), np.concatenate(y_preds))
    rmse = mean_squared_error(np.concatenate(ys), np.concatenate(y_preds), squared=False)
    if config.wandb_log:
        wandb.log({"r2": r2, "rmse": rmse}, step=config['samples_seen'])

    metrics_dict = {'r2': r2, 'rmse': rmse, 'loss': test_loss}

    return metrics_dict


def build_datasets(config:ConfigDict) -> Tuple[BatchedFeaturesDataset, BatchedFeaturesDataset]:
    """
    Build the train and validation datasets to be used for training the loss prediction model
    """
    loss_files = glob.glob(os.path.join(config.loss_dir, '*.json'))
    dfs = [pd.DataFrame.from_dict(json.load(open(f, 'r')), orient='index') for f in loss_files]
    loss_df = pd.concat(dfs, axis=0).reset_index(drop=True).assign(batch_size=lambda df: df.image_ids.apply(len))
    loss_df = pd.concat([loss_df.drop(columns=['loss_dict']), loss_df.loss_dict.apply(pd.Series)], axis=1)
    features = torch.load(config.feature_pt_file)

    # transformation to handle the batched losses
    transform = T.Compose([
        # apply either AvgFeaturesTransform or RandomSampleFeaturesTransform randomly
        # AvgFeaturesTransform will average the features of all samples in the batch
        # RandomSampleFeaturesTransform will randomly select the features of one of the samples in the batch
        RandomSelect(AvgFeaturesTransform(), RandomSampleFeaturesTransform(), p=0.6),
        # get rid of the batch dimension
        lambda x: x.squeeze(0),
    ])
    # we sample some images to use as test set and make sure they are not included
    # in any of the batches in the training set to avoid data leakage
    test_image_ids = loss_df.image_ids.explode().drop_duplicates().sample(frac=0.1, random_state=42).tolist()
    y_test = loss_df[loss_df.image_ids.apply(lambda l: any(i in test_image_ids for i in l))]
    y_train = loss_df[~loss_df.image_ids.apply(lambda l: any(i in test_image_ids for i in l))]
    
    # make the datasets:
    features = features.detach() # detach the features from the graph
    features_dict = {i : features[i] for i in range(features.size(0))}
    train_ds = BatchedFeaturesDataset(features_dict, y_train, ['loss','loss_ce', 'batch_size'], transform=transform)
    test_ds = BatchedFeaturesDataset(features_dict, y_test, ['loss','loss_ce', 'batch_size'], transform=transform)
    
    return train_ds, test_ds
        
def build_model(config:ConfigDict) -> DETRLossPredictor:
    """
    Build the loss prediction model
    """
    feature_size = config.feature_size
    hidden_size = config.hidden_size

    model = DETRLossPredictor(feature_size, hidden_size, output_size=1)
    return model


def wandb_init(project, model, **kwargs):
    import wandb
    wandb.init(project=project, config=kwargs)
    wandb.watch(model)

if __name__ == '__main__':
    import argparse
    from bacili_detection.utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='name of config', default=DEFAULT_CONFIG)
    args = parser.parse_args()
    
    config = load_config(args.config, configs_module=CONFIG_MODULE)
    main(config)