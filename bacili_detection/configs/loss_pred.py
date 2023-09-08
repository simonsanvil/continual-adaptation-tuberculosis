from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict


class DETRLossPredictionExperimentConfig(BaseModel):
    batch_size: int = Field(1, description="Batch size")
    num_epochs: int = Field(100, description="Number of epochs")
    stop_patience: int = Field(7, description="Number of epochs to wait before early stopping")
    lrs_patience: int = Field(3, description="Number of epochs to wait before reducing the learning rate")
    eval_every: int = Field(5, description="Evaluate the model every x epochs and print loss/metrics")
    device: str = Field("cpu", description="Device to use for training")
    target_loss: str = Field('loss_ce', description="The loss to predict")
    model_save_dir: str = Field('', description="Where to save the model")
    lr: float = Field(0.001, description="Learning rate")
    log_wandb: bool = Field(False, description="Whether to log to wandb")
    wandb_project: str = Field('bacilli-detection', description="Wandb project name")