import torch
from typing import Callable

def box_iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
    """
    # box1 = box1.transpose(1,0)
    # box2 = box2.transpose(1,0)
  
    # Calculate the (x1, y1) and (x2, y2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = torch.max(box1[0], box2[0])
    yi1 = torch.max(box1[1], box2[1])
    xi2 = torch.min(box1[2], box2[2])
    yi2 = torch.min(box1[3], box2[3])
    inter_area = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = (box1_area + box2_area) - inter_area
  
    # compute the IoU
    iou = inter_area / union_area

    return iou

def evaluate_prediction(prediction, target, iou_thresh):
    evaluated = torch.zeros(target.shape[0])
    detected = torch.zeros(prediction.shape[0])
    precision = 0
    # base cases
    if target.shape[0] == 0 and prediction.shape[0] == 0:
        return 1, 1
    elif target.shape[0] == 0 and prediction.shape[0] > 0:
        return 0, 0
    elif target.shape[0] > 0 and prediction.shape[0] == 0:
        return 0, 0
    
    for i in range(prediction.shape[0]): # for each predicted box
        # calculate the iou with each target box
        ious = torch.zeros(target.shape[0])
        for j in range(target.shape[0]):
            ious[j] = box_iou(prediction[i], target[j])
        best_gt_idx = torch.argmax(ious)
        if ious[best_gt_idx] > iou_thresh:
            if evaluated[best_gt_idx] == 0:
                precision += 1
                evaluated[best_gt_idx] = 1
                detected[i] = 1
    recall = detected.sum() / target.shape[0]
    precision = precision / prediction.shape[0]
    return precision, recall


def compute_ap(model, inference_function: Callable, dataloader, iou_thresh: float):
    """
    Single-class average precision (AP) for a given IoU threshold
    """
    precisions = []
    for (x, target) in dataloader:
        predictions = inference_function(model, x)
        precision, _ = evaluate_prediction(predictions, target['boxes'], iou_thresh)
        precisions.append(precision)

    # print(f"AP@{iou_thresh*100:.0f}: {torch.mean(torch.Tensor(precisions)):.4f}")
    return torch.mean(torch.Tensor(precisions))

def compute_ar(model, inference_function: Callable, dataloader, iou_thresh: float):
    """
    Single-class average recall (AR) for a given IoU threshold
    """
    recalls = []
    for (x, target) in dataloader:
        predictions = inference_function(model, x)
        _, recall = evaluate_prediction(predictions, target['boxes'], iou_thresh)
        recalls.append(recall)
    
    # print(f"AR@{iou_thresh*100:.0f}: {torch.mean(torch.Tensor(recalls)):.4f}")
    return torch.mean(torch.Tensor(recalls))

def compute_precision_recall(model, inference_function: Callable, dataloader, iou_thresh: float):
    """
    Single-class average precision (AP) for a given IoU threshold
    """
    precisions = []
    recalls = []
    for (x, target) in dataloader:
        predictions = inference_function(model, x)
        precision, recall = evaluate_prediction(predictions, target['boxes'], iou_thresh)
        precisions.append(precision)
        recalls.append(recall)

    # print(f"AP@{iou_thresh*100:.0f}: {torch.mean(torch.Tensor(precisions)):.4f}")
    return torch.mean(torch.Tensor(precisions)), torch.mean(torch.Tensor(recalls))