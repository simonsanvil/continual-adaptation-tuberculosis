import os, sys, time
from typing import Any, List, Dict, Tuple, Literal, TypedDict, Union
from PIL import Image

import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from sqlalchemy.orm import joinedload


from annotations import db
from annotations.object_detection.object_detection import ImageForObjectDetection, Rect
from .preprocessing import mask_filter, tile_coords

ODAnnotationType = TypedDict("DatasetOutput", {
    "image": Image.Image,
    "rects": List[Rect],
    "labels": List[str],
    "image_id": int,
    "area": List[float],
})
        

class BaciliDataset:
    """
    Dataset for the bacili detector
    """
    IMG_WIDTH = 1632
    IMG_HEIGHT = 1224 
    KERNEL_SIZE=80 # size of the sliding window (1:1 aspect ratio)
    STRIDE=40 # 40 pixels between sliding windows
    # MASK_FILTER_THRESHOLD is how many pixels can be missing
    # from a candidate box after masking
    # to include it in the dataset
    MASK_FILTER_THRESHOLD = 60*60*3 - 10 

    def __init__(self, *, train:bool=False, tags:Union[str, List]=None, filter_threshold:int=None):
        self.tags = tags
        self.filter_threshold = filter_threshold
        self.train = train
        self.dataset = None # lazy load

    def __len__(self):
        if self.dataset is None:
            self.load()
        return len(self.dataset)
    
    def load(self):
        self.dataset = self.load_dataset(self.tags, self.filter_threshold)

    def pytorch(self):
        from .pytorch import BaciliDatasetPytorch
        return BaciliDatasetPytorch(self.dataset, self.train)

    @classmethod
    def load_dataset(
            cls,
            tags:Union[str, List]=None,
            filter_threshold:int=None,
        ) -> List[ODAnnotationType]:
        """
        Load the dataset from the annotations
        Returns a list of dictionaries of the form:
        {
            "image": PIL.Image,
            "rects": List[Tuple[float, float, float, float]],
            "image_id": int,
            "labels": List[str]
            "area": List[float],
        }
        """
        tags = tags or ["train"]
        th = filter_threshold or cls.MASK_FILTER_THRESHOLD
     
        if isinstance(tags, str):
            tags = [tags]
        with db.get_session(os.environ['DB_CONN_STR']) as session:
            artifacts = session.query(
                db.Artifact
            ).join(
                db.ArtifactTag
            ).where(
                db.ArtifactTag.tag.in_(tags)
            ).options(
                joinedload(db.Artifact.annotations)
            ).options(
                joinedload(db.Artifact.annotations)
                .joinedload(db.Annotation.properties)
            ).all()

        tiled_coords, nr, nc = tile_coords(cls.IMG_WIDTH, cls.IMG_HEIGHT, cls.KERNEL_SIZE, cls.STRIDE)
        dataset = []
        for artifact in artifacts:
            if len(artifact.annotations)==0:
                continue
            imod = ImageForObjectDetection.from_db(artifact)
            img = imod.numpy()
            # imgs.append(img)
            mask = mask_filter(img)
            masked_img = img * mask.reshape(*mask.shape, 1)
            candidate_boxes = np.array([
                (x1, y1, x2, y2)
                for x1, y1, x2, y2 in tiled_coords
                if np.sum(masked_img[x1:x2, y1:y2]==0) < th
            ]).astype(np.int16)
            # how many missing true rects?
            centers = [r.center for r in imod.rects]
            overlaps = np.array([
                ((candidate_boxes[..., 0] < center[0]) & \
                (candidate_boxes[..., 1] < center[1]) & \
                    (candidate_boxes[..., 2] > center[0]) & \
                        (candidate_boxes[..., 3] > center[1]))
                for center in centers
            ])
            missing_count = sum(overlaps.sum(axis=1)==0)
            if overlaps.shape[0]==1:
                overlaps = overlaps.squeeze()
            else:
                overlaps = overlaps.any(axis=0)
            labels = np.zeros(len(candidate_boxes))
            labels[overlaps] = 1 # 1 for bacili, 0 for background
            # candidate_rects = [Rect.from_bbox(box) for box in candidate_boxes] 
            img_data_dict = {
                "image": imod.pil(),
                "image_id": imod.artifact.id,
                "image_uri": imod.uri,
                "num_candidate_boxes": nr*nc,
                "num_filtered_boxes": len(candidate_boxes),
                "num_true": len(imod.rects),
                "num_missing_true": missing_count,
                "boxes": candidate_boxes,
                "labels": labels.tolist()
            }
            dataset.append(img_data_dict)
                
        return dataset