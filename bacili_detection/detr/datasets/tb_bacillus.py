import os, dotenv, sys
from typing import Any, Dict, List, Literal
from pathlib import Path
from PIL import Image

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy.orm import joinedload

from annotations.object_detection.object_detection import ImageForObjectDetection, Rect
from annotations.object_detection.dataset import DatasetForObjectDetection
from annotations import db

from . import transforms as T

sys.path.append('bacili_detection/src')

class TBBacilliDataset(DatasetForObjectDetection):

    def __init__(self, 
            artifact_tags:list, *, 
            train:bool=False, 
            box_format:str='xyxy', 
            transform:list=None,
            class_name:str="TBbacillus",
            image_dir:str=None,
            **kwargs
        ):
        if isinstance(artifact_tags, str):
            artifact_tags = [artifact_tags]
        if isinstance(artifact_tags[0], db.Artifact):
            artifacts = artifact_tags
        else:
            dotenv.load_dotenv('.env')
            session = db.get_session(os.environ.get("DATABASE_URI"))
            artifacts = get_artifacts_with_tags(artifact_tags, session)
        self._artifact_subsets = {}
        if len(artifact_tags) > 1:
            for tag in artifact_tags:
                self._artifact_subsets[tag] = [art for art in artifacts if any(tag==t.tag for t in art.tags)]

        odimages = [ImageForObjectDetection.from_db(art, img_dir=image_dir) for art in artifacts]
        class_mapping = {"N/A":0, class_name:1}
        self._transform = transform
        self._image_dir = image_dir
        super().__init__(odimages, train=train, box_format=box_format, class_mapping=class_mapping, transform=None, **kwargs)
        # self.transform_includes_target = True # for pytorch
        self.output_format('tuple')
        self.torch()

    def subset(self, tag:str):
        if tag not in self._artifact_subsets:
            raise ValueError(f"Unknown subset tag: {tag}")
        return TBBacilliDataset(self._artifact_subsets[tag], train=self.train, transform=self._transform, image_dir=self._image_dir)

    def __getitem__(self, idx: int) -> Any:
        img, target = super().__getitem__(idx)
        # convert every value in target to torch.tensor and remove those that are type str
        target['labels'] = target['labels'].type(torch.int64)
        # target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        # target['area'] = torch.tensor(target['area'], dtype=torch.float32)
        if target['boxes'].shape[0] == 0:
            # make sure the boxes tensor is not empty and has shape (-1,4)
            target['boxes'] = torch.zeros((0,4), dtype=torch.float32)
        target['size'] = target['size'].type(torch.int64)
        target['orig_size'] = target['orig_size'].type(torch.int64)
        target['image_id'] = torch.tensor(target['image_id'], dtype=torch.int64)
        target['image_db_id'] = torch.tensor(target['image_db_id'], dtype=torch.int64)
        # target['is_crowd'] = torch.zeros((len(target['labels']),), dtype=torch.int64)
        # apply transform
        if self._transform is not None:
            img, target = self._transform(img, target)
        return img, target

    def postprocess(self, img:torch.Tensor, target:dict):
        """ 
        Unnormalize the image and target boxes
        """
        img = self.unnormalize(img)
        target['boxes'] = self.unnormalize_boxes(target['boxes'], img.size)
        target['labels'] = [self.id2label[l.item()] for l in target['labels']]
        return img, target
    
    def unnormalize(self, img:torch.Tensor):
        """ 
        Unnormalize the image
        """
        img = img.clone()
        img = img.squeeze(0)
        img = img.permute(1,2,0)
        img = img.cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = img * 255
        img = Image.fromarray(img.astype(np.uint8))
        return img
    
    def unnormalize_boxes(self, boxes:torch.Tensor, img_size:tuple):
        """ 
        Unnormalize the target boxes
        """
        x_c, y_c, w, h = boxes.unbind(1)
        width, height = img_size
        b_xyxy = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                 (x_c + 0.5 * w), (y_c + 0.5 * h)]
        b_xyxy = torch.stack(b_xyxy, dim=1)
        b_xyxy[:, 0::2] *= width
        b_xyxy[:, 1::2] *= height
        return b_xyxy.type(torch.int16).cpu().numpy()

def make_ds_transforms(tag:str):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [672, 704, 736, 768, 800]
    if tag == "train":
        train_transforms = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.4, 0.4, saturation=0, hue=0),
            T.RandomResize(scales, max_size=1333),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=1333),
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 600),
            #         T.RandomResize(scales, max_size=1333),
            #     ])
            # ),
            normalize,
        ])
        return train_transforms
    else:
        eval_transforms = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
        return eval_transforms


def get_artifacts_with_tags(tags:List[str], db_session, project_name:str="Bacilli Detection") -> list:
    artifacts = db_session.query(db.Artifact)\
        .join(db.ArtifactTag)\
        .join(db.Project).filter(
            db.Project.name == project_name,
            db.ArtifactTag.tag.in_(tags)
        ).options(
            joinedload(db.Artifact.annotations)
        ).options(
            joinedload(db.Artifact.annotations)
            .joinedload(db.Annotation.properties)
        ).order_by(db.Artifact.id)\
        .all()
    return artifacts

def build(ds_set:Literal["train","val"], args=None) -> TBBacilliDataset:
    if ds_set == "train":
        tags = ["train"]
    elif ds_set == "val":
        tags = ["val"]
    elif ds_set == "test":
        tags = ["test"]
    else:
        raise ValueError(f"Unknown dataset set: {ds_set}")
    ds = TBBacilliDataset(tags, train=(ds_set=="train"), transform=make_ds_transforms(ds_set), image_dir=args.image_dir)
    return ds
