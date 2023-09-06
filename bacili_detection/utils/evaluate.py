from typing import Callable, Dict, List, Tuple
from shapely import geometry
from shapely import ops
from tqdm import tqdm

import numpy as np
from annotations.object_detection.dataset import DatasetForObjectDetection

# calculate the IoU between the predicted and the GT boxes
# as well as the precision and recall for each image

def evaluate(
        dataset:DatasetForObjectDetection, 
        inference_func:Callable[[List,object],List[Tuple[float, float, float, float]]],
        batch_size:int=1,
    ) -> Dict[str, List[float]]:
    """
    Evaluate a model on a dataset
    calculate the IoU between the predicted and the GT boxes
    as well as the precision and recall for each image
    """
    metrics = dict(
        image_id = [],
        precision = [],
        recall = [],
        iou = [],
    )
    for i in tqdm(range(0, len(dataset), batch_size)):
        images = dataset._images[i:i+batch_size]
        imgs = [im.pil() for im in images]
        bboxes = inference_func(imgs)
        for i, img in enumerate(images):
            pred_boxes = bboxes[i]
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
    return metrics