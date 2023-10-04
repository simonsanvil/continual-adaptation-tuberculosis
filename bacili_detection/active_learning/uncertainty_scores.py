from typing import Iterable
import torch
from torch.utils.data import DataLoader
from bacili_detection.detr.models import detr_inference
from bacili_detection.utils import merge_rects
from bacili_detection.detr.datasets.tb_bacillus import TBBacilliDataset
from annotations.object_detection.rect import Rect
import numpy as np

def get_margin_uncertainty_scores(dataloader, model):
    model.eval()
    uncertainties = []

    with torch.no_grad():
        for i, (input, _ ) in enumerate(dataloader):
        
            output = model(input)
            
            probs = output['pred_logits'].softmax(dim=-1)

            if probs.shape[1] < 2: #if less than two classes present skip image.
                continue

            # calculate margin uncertainty (difference between top two probabilities)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin_uncertainty = sorted_probs[:, 0] - sorted_probs[:, 1]

            uncertainties.append((i, margin_uncertainty.item()))

    # Sort images based on uncertainty from high to low
    uncertainties.sort(key=lambda x: x[1], reverse=False)
    return uncertainties

def merged_area_scores(dataloader, model):
    model.eval()
    scores = np.zeros(len(dataloader))

    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
        
            output = model(input)
            
            probs = output['pred_logits'].softmax(-1)[0, :, :-1]

            if probs.shape[1] < 2: #if less than two classes present skip image.
                continue

            # calculate margin uncertainty (difference between top two probabilities)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin_uncertainty = sorted_probs[:, 0] - sorted_probs[:, 1]
            top = margin_uncertainty.argsort()[-20:]
            boxes_top = output['pred_boxes'][0, top].cpu()
            rects = []
            for j, box in enumerate(boxes_top):
                xc, yc, w, h = box
                imh, imw = target[0]['orig_size']
                box = [xc*imw, yc*imh, w*imw, h*imh]
                box = [b.item() for b in box]
                rects.append(Rect.from_bbox(box, bbox_format='cxcywh'))
            scores[i] = merged_area_ratio(rects)
        return scores


    # Sort images based on uncertainty from high to low
    # scores.sort(key=lambda x: x[1], reverse=False)
    return scores

def merged_area_ratio(boxes):
    """
    Calculates the non overlapping area of a set of boxes.
    """
    # calculate the total area of the boxes
    # boxes = [box.shapely() for box in boxes]
    from shapely.geometry import box as shapely_box
    from shapely.ops import cascaded_union
    from shapely.geometry import MultiPolygon

    boxes = [shapely_box(*box.xyxy) for box in boxes]
    merged = MultiPolygon(boxes).buffer(0)
    return merged.area / sum([box.area for box in boxes])