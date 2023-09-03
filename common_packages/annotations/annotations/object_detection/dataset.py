from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
from .object_detection import ImageForObjectDetection, Rect, Rects
from ..db import Artifact

class DatasetForObjectDetection:

    def __init__(self, 
            images:List[Union[ImageForObjectDetection, str]], 
            *, 
            box_format:str='xyxy',
            train:bool=False, 
            transform=None,
            class_mapping:Dict[str, int]=None,
            info:Dict[str, Any]=None,
            output_format:Literal['odimg','tuple','dict']='odimg',
        ):
        self.train = train
        self.info = info or {}
        images_ = []
        for img in images:
            if isinstance(img, str):
                img = ImageForObjectDetection(img)
            assert isinstance(img, ImageForObjectDetection), f"Expected ImageForObjectDetection, got {type(img)}"
            images_.append(img)
        self.transform = transform
        self._images = images_
        self._box_format = box_format
        self._array_type = np.array
        self._output_format = output_format
        self.class_mapping = class_mapping
        self.id2label = {v:k for k,v in self.class_mapping.items()}
        self.transform_includes_target = False
    
    @classmethod
    def from_db(cls, artifacts:List[Artifact], **kwargs) -> 'DatasetForObjectDetection':
        images = [ImageForObjectDetection.from_db(art) for art in artifacts]
        return cls(images, **kwargs)

    def __len__(self):
        return len(self.images)
    
    def __iter__(self):
        for img in self.images:
            yield img

    def __add__(self, other):
        return DatasetForObjectDetection(
            self._images + other._images, 
            train=self.train, 
            transform=self.transform,
            class_mapping=self.class_mapping,
            info=self.info,
            box_format=self._box_format,
            output_format=self._output_format,
        )
    
    def __getitem__(self, idx:int) -> Any:
        img = self._images[idx]
        if self._output_format == 'odimg':
            return img
        boxes = [self._rect_to_format(r) for r in img.rects]
        labels = [r.label for r in img.rects]
        areas = [r.area for r in img.rects]
        img = img.pil()
        w , h = img.size
        size = [int(h), int(w)]
        if self._array_type is not None:
            boxes = self._array_type(boxes)
            areas = self._array_type(areas)
            size = self._array_type(size)
        if self.class_mapping is not None:
            labels = [self.class_mapping[l] for l in labels]
            if self._array_type is not None:
                labels = self._array_type(labels)
        target = {
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'size': size,
            'orig_size': size,
            'image_id': idx,
            'image_db_id': self._images[idx].artifact.id,
        }
        if self.transform is not None:
            if isinstance(self.transform, dict):
                # if transform is a dict is expected that its keys include one of:
                # 'image', 'boxes', 'labels', 'area'
                # and the values are callables
                if 'image' in self.transform:
                    img = self.transform['image'](img)
                if 'boxes' in self.transform:
                    target['boxes'] = self.transform['boxes'](target['boxes'])
                if 'labels' in self.transform:
                    target['labels'] = self.transform['labels'](target['labels'])
                if 'area' in self.transform:
                    target['area'] = self.transform['area'](target['area'])
            else:
                # default that it will be applied to the image
                if self.transform_includes_target:
                    img, target = self.transform(img, target)
                else:
                    img = self.transform(img)
        if self._output_format == 'tuple':
            return img, target
        elif self._output_format == 'dict':
            target['image'] = img
            return target
        else:
            raise ValueError(f"Unknown output format {self._output_format}")
    

    def _rect_to_format(self, rect:Rect):
        if self._box_format == 'xywh':
            return rect.xywh
        elif self._box_format == 'xyxy':
            return rect.xyxy
        elif self._box_format == 'cxcywh':
            return rect.cxcywh
        else:
            raise ValueError(f"Unknown box format {self._box_format}")
    
    @property
    def images(self):
        if self._output_format == 'odimg':
            return self._images
        else:
            return [img.pil() for img in self._images]
    
    def display(self, nrows=1, ncols=1, figsize=(10,10), **kwargs):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, ax in enumerate(axs.flat):
            self._images[i].display(ax=ax, **kwargs)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def box_format(self, box_format:Literal['xywh', 'xyxy', 'cxcywh']):
        self._box_format = box_format

    def numpy(self):
        self._array_type = np.array
        return self

    def torch(self, **kwargs):
        from torch import tensor
        self._array_type = partial(tensor, **kwargs)
        return self
    
    def tf(self, **kwargs):
        from tensorflow import Tensor
        self._array_type = partial(Tensor, **kwargs)
        return self
    
    def output_format(
            self, 
            out_format:Union[Literal['odimg','dict', 'tuple'], object]=None,
            array_type:Union[Literal['np','torch','tf'], Callable]=None,
        ):
        if out_format == tuple:
            out_format = 'tuple'
        if out_format == dict:
            out_format = 'dict'
        if out_format == ImageForObjectDetection:
            out_format = 'odimg'
        if out_format is not None:
            self._output_format = out_format
        if array_type == 'np':
            self._array_type = np.array
        elif array_type == 'torch':
            from torch import tensor
            self._array_type = tensor
        elif array_type == 'tf':
            from tensorflow import Tensor
            self._array_type = Tensor
        else:
            self._array_type = array_type

    def to_coco(self, as_dict:bool=False):
        """
        Convert to coco dataset format:
        COCO dataset must be a JSON dict with the following keys:
        - annotations: list of dicts with keys:
            - id: int
            - image_id: int
            - bbox: [x,y,width,height]
            - area: float
            - category_id: int
            - iscrowd: 0 or 1
        - images: list of dicts with keys:
            - id: int
            - width: int
            - height: int
            - file_name: str
        - categories: list of dicts with keys:
            - id: int
            - name: str
        - info (optional): dict with keys:
            - description: str
            - url: str
            - version: str
            - year: int
            - contributor: str
            - date_created: datetime
        """ 
        annotations = []
        images = []
        categories = []
        for i, img in enumerate(self._images):
            images.append({
                'id': i,
                'width': img.width,
                'height': img.height,
                'file_name': img.uri,
            })
            for j, rect in enumerate(img.rects):
                annotations.append({
                    'id': j,
                    'image_id': i,
                    'bbox': rect.xywh,
                    'area': rect.area,
                    'category_id': self.class_mapping[rect.label],
                    'iscrowd': 0,
                })
            if self.class_mapping is not None:
                for k, v in self.class_mapping.items():
                    categories.append({
                        'id': v,
                        'name': k,
                    })
        ds = {
            'annotations': annotations,
            'images': images,
            'categories': categories,
            'info': self.info,
        }
        if not as_dict:
            from pycocotools.coco import COCO
            coco = COCO()
            coco.dataset = ds
            coco.createIndex()
            return coco
        return ds
    