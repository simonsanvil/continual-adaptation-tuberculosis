# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Union
from .detr import build
import torch
import numpy as np
from PIL import Image

def build_model(args):
    return build(args)

def detr_inference(
        imgs:Union[Image.Image,List[Image.Image]], 
        model, 
        transform:callable, 
        id2label:dict=None, 
        threshold:float=0.7,
        device:str='cpu',
    ):
    with torch.no_grad():
        # inputs = processor(images=img, return_tensors="pt")['pixel_values']
        # inputs = torchvision.transforms.functional.to_tensor(img)#.unsqueeze(0)
        if not isinstance(imgs, list):
            imgs = [imgs]
        inputs = [transform(i).unsqueeze(0) for i in imgs]
        inputs = torch.cat(inputs, dim=0) # batched inference
        inputs = inputs.to(device)
        model = model.to(device)
        outputs = model(inputs)
        # keep only predictions with 0.7+ confidence
        keep = outputs['pred_logits'].softmax(-1)[0, :, :-1].max(-1).values > threshold
        # convert to numpy
        x_c, y_c, w, h = outputs['pred_boxes'][0, keep].cpu().numpy().T
        # convert from xcenter, ycenter, width, height to xyxy (corner) format
        bboxes = np.stack(
            [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        ).transpose(1, 0)
        # rescale boxes to original image (since DETR is scaled to 800x800)
        for img, bbox in zip(imgs, bboxes):
            bboxes = bboxes * np.array([img.width, img.height, img.width, img.height])
        # the output format of detr is 
        bboxes = bboxes.astype(np.int16)
        # get labels
        labels = outputs['pred_logits'].softmax(-1)[0, keep].argmax(-1).cpu().numpy()
        if id2label:
            labels = [id2label[label] for label in labels]
    return bboxes, labels