# 5. build a DNN that takes the features as input and outputs the loss
from typing import Dict
import json

# to define torch transform
import torch
from torch import nn
from torchvision import transforms as T

# calculate the features from DETR
import pandas as pd
from bacili_detection.detr.util.features import calculate_features


class DETRLossPredictor(nn.Module):
    """
    A simple DNN that takes the features from the CNN backbone of DETR as input and outputs the loss
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # input_size = 256x25x34
        self.conv = nn.Conv2d(input_size, 256, kernel_size=25, stride=1, padding=0)
        # if the input is 256x25x34 with kernel_size=25, stride=1, padding=0, 
        # then the output is 256x1x10 since (34-25)/1 + 1 = 10
        # where 34 is the width of the image and 25 is the kernel size
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256*10, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
    @torch.no_grad()
    def predict(self, x:dict):
        return self.forward(x).detach().numpy()
    
    @torch.no_grad()
    def predict_from_image(self, image, model, transform, device='cpu'):
        src, pos = calculate_features([image], model, transform, device=device)
        return self.predict(src)
    

