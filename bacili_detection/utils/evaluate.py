from typing import Callable, Dict, List, Tuple, Union
from shapely import geometry
from functools import partial
from tqdm import tqdm
# from shapely import ops

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T

from annotations.object_detection.dataset import DatasetForObjectDetection
from bacili_detection.detr.datasets.tb_bacillus import TBBacilliDataset
from bacili_detection.detr.util.metrics import evaluate_prediction
from bacili_detection.detr.models import detr_inference


def evaluate_trained_model(checkpoint_dir:str, file:str='eval.csv', device:str='cpu', image_dir=''):
    # should receive the experiment params as  arguments
    # load the model from checkpoint
    checkpoint_path = f'{checkpoint_dir}/checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    m = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', 
    pretrained=False, num_classes=2)
    m.load_state_dict(checkpoint['model'])
    m.eval();
    print("model loaded at epoch: ", checkpoint['epoch'])
    test_dataset = TBBacilliDataset(['test'], transform=None, image_dir=image_dir)
    transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    detr_inference_func = partial(
        detr_inference, 
        model=m, transform=transform, 
        id2label=test_dataset.id2label, 
        threshold=0.95, labels=False, 
        device=device
    )
    metrics = evaluate(test_dataset, detr_inference_func, batch_size=1, iou_thresholds=[0.3, 0.5])
    metrics_df = pd.DataFrame(metrics)
    if file is not None:
        if file.endswith('.csv'):
            metrics_df.to_csv(f'{file}', index=False)
        elif file.endswith('.json'):
            metrics_df.to_json(f'{file}', orient='records')
    return metrics_df

def evaluate(
        dataset:DatasetForObjectDetection, 
        inference_func:Callable[[List[object]],List[Tuple[float, float, float, float]]],
        batch_size:int=1,
        iou_thresholds:Union[float, List[float]]=0.5,
    ) -> Dict[str, List[float]]:
    """
    Evaluate a model on a dataset
    calculate the IoU between the predicted and the GT boxes
    as well as the precision and recall for each image
    """
    iou_thresholds = [iou_thresholds] if isinstance(iou_thresholds, float) else iou_thresholds
    metrics = dict(
        image_id = [],
        precision = [],
        recall = [],
        iou = [],
    )
    for iou_thresh in iou_thresholds:
        metrics[f"p_at_{iou_thresh*100:.0f}"] = []
        metrics[f"r_at_{iou_thresh*100:.0f}"] = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        images = dataset._images[i:i+batch_size]
        imgs = [im.pil() for im in images]
        bboxes = inference_func(imgs)
        for i, img in enumerate(images):
            pred_boxes = bboxes[i]
            gt_xyxy = [rect.xyxy for rect in img.rects]
            if len(pred_boxes) == 0 and len(img.rects) == 0:
                iou, precision, recall = 1, 1, 1
            elif len(pred_boxes) == 0 and len(img.rects) > 0:
                iou, precision, recall = 0, 0, 0
            elif len(pred_boxes) > 0 and len(img.rects) == 0:
                iou, precision, recall = 0, 0, 0
            else:
                # get the GT boxes
                gt_boxes = [geometry.box(*rect.xyxy) for rect in img.rects]
                # calculate the IoU between the predicted and the GT boxes
                gt_poly = geometry.MultiPolygon(gt_boxes).buffer(0)
                pred_poly = geometry.MultiPolygon(geometry.box(*b) for b in pred_boxes).buffer(0)
                iou = gt_poly.intersection(pred_poly).area / gt_poly.union(pred_poly).area
                # calculate the precision and recall
                if not hasattr(pred_poly, 'geoms'):
                    pred_poly = geometry.MultiPolygon([pred_poly])
                if not hasattr(gt_poly, 'geoms'):
                    gt_poly = geometry.MultiPolygon([gt_poly])
                # prediction confidence is calculated as the distance between the centroid of the predicted box
                # and the centroid of the GT box iff the GT box is inside any of the predicted boxes
                preds = np.array([
                    any(pred.contains(gt.centroid) or pred.distance(gt.centroid) < 5 
                            for gt in gt_poly.geoms)
                    for pred in pred_poly.geoms
                ])
                precision = preds.sum() / len(preds)
                recall = preds.sum() / len(gt_boxes)

            metrics['image_id'].append(img.name)
            metrics['iou'].append(iou)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            for iou_thresh in iou_thresholds:
                p_at, r_at = prec_recall_at_iou(gt_xyxy, pred_boxes, iou_thresh)
                metrics[f"p_at_{iou_thresh*100:.0f}"].append(p_at)
                metrics[f"r_at_{iou_thresh*100:.0f}"].append(r_at)

    return metrics

def prec_recall_at_iou(
        trues:List[Tuple[float,float,float,float]], 
        preds:List[Tuple[float,float,float,float]],
        iou_thresh:float=0.5,
    ):
    """
    Calculate the precision and recall for a single image
    """
    import torch
    trues = torch.tensor(trues)
    preds = torch.tensor(preds)
    precision, recall = evaluate_prediction(preds, trues, iou_thresh)
    if isinstance(precision, torch.Tensor):
        precision = precision.item()
    if isinstance(recall, torch.Tensor):
        recall = recall.item()
    return precision, recall