# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Union
from .detr import build
import torch
import numpy as np
from PIL import Image

def build_model(args):
    return build(args)

@torch.no_grad()
def detr_inference(
        imgs:Union[Image.Image,List[Image.Image]], 
        model, 
        transform:callable, 
        id2label:dict=None, 
        threshold:float=0.7,
        device:str='cpu',
        labels:bool=True,
    ):
    """
    Perform inference with DETR on a list of images
    """
    # inputs = processor(images=img, return_tensors="pt")['pixel_values']
    # inputs = torchvision.transforms.functional.to_tensor(img)#.unsqueeze(0)
    model.eval()
    if not isinstance(imgs, list):
        imgs = [imgs]
    inputs = [transform(im).unsqueeze(0) for im in imgs]
    inputs = torch.cat(inputs, dim=0) # batched inference
    inputs = inputs.to(device)
    model = model.to(device)
    outputs = model(inputs)
    return_labels = labels
    # keep only predictions with confidence > threshold
    keep = outputs['pred_logits'].softmax(-1)[:, :, :-1].max(-1).values > threshold
    # denormalize bboxes
    x_c, y_c, w, h = outputs['pred_boxes'].detach().transpose(0, 2)
    bboxes = torch.stack(
        [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    ).permute(2, 1, 0)
    # to cpu
    bboxes = bboxes.cpu()
    keep = keep.cpu()
    # filter out low confidence bboxes
    boxes_per_img = []
    labels_per_img = []
    for i, img in enumerate(imgs):
        labels = outputs['pred_logits'][i, keep[i]].argmax(-1)
        im_bboxes = bboxes[i, keep[i]] * torch.tensor([img.width, img.height, img.width, img.height])
        im_bboxes = im_bboxes.numpy().astype(np.int16)
        if id2label is not None:
            labels = [id2label[label.item()] for label in labels]
        else:
            labels = [label.item() for label in labels]
        boxes_per_img.append(im_bboxes)
        labels_per_img.append(labels)
    
    if return_labels:
        return boxes_per_img, labels_per_img
    return boxes_per_img