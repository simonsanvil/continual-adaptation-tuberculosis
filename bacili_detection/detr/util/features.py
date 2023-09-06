"""
Build features to predict the loss of a DETR model on a given dataset
"""

from typing import Dict, List
from tqdm.notebook import tqdm

import torch

from bacili_detection.detr.util.misc import nested_tensor_from_tensor_list
from bacili_detection.detr.models.detr import DETR
# from bacili_detection.detr.models import detr_inference
from dataclasses import dataclass

@dataclass
class DETRFeatures:
    backbone: Dict[str, torch.Tensor] = None
    encoder_memory: torch.Tensor = None
    transformer: torch.Tensor = None
    output: Dict[str, torch.Tensor] = None

    def to(self, device:str, stage='all'):
        """
        Move the features to a given device
        """
        if self.backbone is not None and (stage=='all' or stage=='backbone'):
            self.backbone['src'] = self.backbone['src'].to(device)
            self.backbone['pos'] = self.backbone['pos'].to(device)
            self.backbone['mask'] = self.backbone['mask'].to(device)
        if self.encoder_memory is not None and (stage=='all' or stage.startswith('encoder')):
            self.encoder_memory = self.encoder_memory.to(device)
        if self.transformer is not None and (stage=='all' or stage == 'transformer'):
            self.transformer = self.transformer.to(device)
        if self.output is not None and (stage=='all' or stage == 'output'):
            self.output['logits'] = self.output['logits'].to(device)
            self.output['boxes'] = self.output['boxes'].to(device)
    

def calculate_features(images:List, model:DETR, transform, device='cpu',stop_at:str=None):
    """
    Calculate the features of a DETR model on a given dataset
    """
    stop_at = stop_at or 'output'
    feat = DETRFeatures()
    model.eval()
    feat.backbone = backbone_features(images, model, transform, device)
    if stop_at == 'backbone' or stop_at==0:
        return feat
    src, pos = feat.backbone['src'], feat.backbone['pos']
    bs, c, h, w = src.shape
    src = src.flatten(2).permute(2, 0, 1)
    pos = pos.flatten(2).permute(2, 0, 1)
    mask = feat.backbone['mask'].flatten(1)
    query_embed = model.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
    with torch.no_grad():
        mem = model.transformer.encoder(src, src_key_padding_mask=mask, pos=pos)
    feat.encoder_memory = mem.permute(1, 2, 0).view(bs, c, h, w)
    if stop_at.startswith('encoder') or stop_at==1:
        return feat
    tgt = torch.zeros_like(query_embed)
    with torch.no_grad():
        hs = model.transformer.decoder(tgt, mem, memory_key_padding_mask=mask,pos=pos, query_pos=query_embed)
    hs = hs.transpose(1, 2)[0]
    feat.transformer = hs
    if stop_at.startswith('transformer') or stop_at==2:
        return feat
    with torch.no_grad():
        out_logits = model.class_embed(hs)
        out_boxes = model.bbox_embed(hs).sigmoid()
    out = {'logits': out_logits[-1], 'boxes': out_boxes[-1]}
    if model.aux_loss:
        with torch.no_grad():
            out['aux_outputs'] = model._set_aux_loss(out_logits, out_boxes)
    feat.output = out
    return feat

# get the features from the CNN backbone of DETR
def backbone_features(images:List, model:DETR, transform, device='cpu'):
    sample = [transform(image) for image in images]
    sample = nested_tensor_from_tensor_list(sample)
    sample = sample.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        features, pos = model.backbone(sample)
        src, mask = features[-1].decompose()
        src, pos = model.input_proj(src), pos[-1]
    return {"src":src, "pos":pos, "mask":mask}

# get the features from the transformer encoder of DETR
def transformer_features(images:List, model:DETR, transform, device='cpu'):
    sample = [transform(image) for image in images]
    sample = nested_tensor_from_tensor_list(sample)
    sample = sample.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        features, pos = model.backbone(sample)
        src, _ = features[-1].decompose()
        src, pos = model.input_proj(src), pos[-1]
        hs = model.transformer(src, mask=None, query_embed=model.query_embed.weight, pos_embed=pos)
    return hs

# get the ecoder-only features from DETR
def encoder_only_features(images:List, model:DETR, transform, device='cpu'):
    sample = [transform(image) for image in images]
    sample = nested_tensor_from_tensor_list(sample)
    sample = sample.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        features, pos = model.backbone(sample)
        src, mask = features[-1].decompose()
        src, pos = model.input_proj(src), pos[-1]
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        query_embed = model.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        mem = model.transformer.encoder(src, mask, query_embed, pos)
    return mem

