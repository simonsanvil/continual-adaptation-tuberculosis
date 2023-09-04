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
        if artifact_tags is None:
            artifact_tags = []
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
        return img, target

    def postprocess(self, img:torch.Tensor, target:dict):
        """ 
        Unnormalize the image and target boxes
        """
        img = self.unnormalize(img)
        target['boxes'] = self.unnormalize_boxes(target['boxes'], img.size)
        target['labels'] = [self.id2label[l.item()] for l in target['labels']]
        return img, target
    
    

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