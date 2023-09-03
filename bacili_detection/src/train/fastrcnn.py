"""
rCNN for Bacili detection using pytorch
"""
import os, sys, time, math
from typing import Any, List, Dict, Tuple, Literal, TypedDict, Union
from tqdm import tqdm
from PIL import Image

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from sqlalchemy.orm import joinedload

import torch.nn as nn

from annotations import db
from annotations.object_detection.object_detection import ImageForObjectDetection, Rect

from ..dataset import BaciliDataset
from . import utils

def train_model(
        num_epochs:int=10,
        *,
        device:str="cpu",
        fine_tune:bool=False,
        lr:float=0.005,
        momentum:float=0.9,
        weight_decay:float=0.0005,
        lr_scheduler_step_size:int=3,
        lr_scheduler_gamma:float=0.1,
        print_freq:int=10,
        checkpoint_path:str=None,
        checkpoint_frequency:int=1,
    ):
    """
    Train the model for object detection using the bacili dataset
    """
    if fine_tune:
        # w/ fine-tuning we will train the head and the classifier layers
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = 2  # 1 class (bacili) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        # w/o fine-tuning we will add a custom backbone
        backbone = torchvision.models.mobilenet_v2(weights='DEFAULT').features
        backbone.out_channels = 1280
        # RPN should generate 80x80 anchors and only one aspect ratio
        anchor_generator = AnchorGenerator(sizes=((80, 80),), aspect_ratios=((1.0, 1.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    # load the data
    train_dataset = BaciliDataset(tag="train", train=True)
    train_dataset.load()
    val_dataset = BaciliDataset(tag="test")
    val_dataset.load()
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        train_dataset.pytorch(), 
        batch_size=2, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    data_loader_test = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, 
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    # move model to the right device
    model.to(device)
    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    # learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
    # train
    for epoch in tqdm(range(num_epochs)):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    # 
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger