""" 
Utility functions for computing losses of the DETR model
"""


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from . import misc as utils
from ..models.detr import SetCriterion, build_matcher

def compute_losses(
        model,
        criterion,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        notebook: bool = False,
        stop_at: int = None,
) -> dict:
    """ 
    Compute the losses of the model on the given data_loader
    """

    losses = {} # dict of id: {'image_ids': [], 'losses': []}
    model.eval()
    criterion.eval()
    model.to(device)
    criterion.to(device)

    tqdm_fn = tqdm_notebook if notebook else tqdm
    counter = 0
    for samples, targets in tqdm_fn(data_loader, total=len(data_loader)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses[counter] = {
            'image_ids': [t['image_id'].item() for t in targets],
            'image_db_ids': [t['image_db_id'].item() for t in targets],
            'loss': loss.item(),
            'loss_dict': {k: v.item() for k, v in loss_dict.items() if k in weight_dict},
        }
        counter += 1
        if stop_at is not None and counter >= stop_at:
            break

    return losses

def build_criterion(args) -> SetCriterion:
    """ 
    Build the criterion for the DETR model
    """
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    return criterion