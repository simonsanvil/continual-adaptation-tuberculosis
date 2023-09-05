"""
Configurations for DETR model experiments
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict

class DatasetConfig(BaseModel):
    tags: List[str] = []

class ModelConfig(BaseModel):
    name: str = Field(..., description="Name of the model")
    backbone: str = Field(..., description="Name of the backbone")
    num_classes: int = Field(..., description="Number of classes")
    pretrained: bool = Field(True, description="Whether to use pretrained weights")
    trainable_backbone_layers: Union[int, str] = Field(
        -1, description="Number of trainable backbone layers"
    )
    trainable_backbone_starting_from: str = Field(
        "backbone.bottom_up.res2", description="Name of the layer to start training from"
    )