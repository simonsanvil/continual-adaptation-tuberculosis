"""
Models to define the properties of an object detection annotation.
"""
import numbers
from pathlib import Path
from typing import List, Optional, IO, Union, ClassVar
from pydantic import BaseModel, Field, PrivateAttr

from .rect import Rect, Rects
from ..db.models import (
    AnnotationProperty, Annotation, Artifact,
    Datastore, ArtifactType, Annotator
)
from .. import db

class ImageForObjectDetection(BaseModel):
    uri: str = Field(..., description="The path to the image.")
    width: Optional[int] = Field(None, description="The width of the image.", gt=0)
    height: Optional[int] = Field(None, description="The height of the image.", gt=0)
    name: Optional[str] = Field(None, description="The name of the image.")
    img_dir: Optional[str] = Field(None, description="The directory containing the image.")
    _artifact: object = PrivateAttr(None)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        image_path = Path(self.uri)
        if self.img_dir is not None:
            image_path = Path(self.img_dir) / image_path
            self.uri = str(image_path)
        # if (self.width is None or self.height is None) and image_path.exists():
        #     import PIL
        #     img = PIL.Image.open(image_path)
        #     self.width, self.height = img.size
        if self.name is None:
            self.name = image_path.stem

    @property
    def artifact(self) -> Artifact:
        """
        The artifact associated with this image.
        """
        return self._artifact
    
    @property
    def rects(self) -> List["Rects"]:
        """
        Get the bounding boxes (rects) for this image.
        """
        rects = []
        for annotation in self.get_annotations():
            rect = Rect.from_bbox(
                bbox=(
                    annotation.get_property("xmin").numeric_value,
                    annotation.get_property("ymin").numeric_value,
                    annotation.get_property("xmax").numeric_value,
                    annotation.get_property("ymax").numeric_value
                ),
                label=annotation.get_property("label").text_value
            )
            rects.append(rect)
        return rects
    
    @property
    def properties(self) -> List[AnnotationProperty]:
        """
        Get the properties for this image.
        """
        return self.artifact.properties
    
    def pil(self, rects=False) -> "PIL.Image":
        """
        Get the PIL image.
        """
        from PIL import Image, ImageDraw

        img_path = Path(self.uri)
        if not img_path.exists():
            raise FileNotFoundError(f"Could not find image at {img_path}")
        im = Image.open(img_path)
        if rects:
            draw = ImageDraw.Draw(im)
            for rect in self.rects:
                draw.rectangle(rect.xyxy, outline="red")
        return im

    def numpy(self) -> "np.ndarray":
        """
        Get the numpy array.
        """
        import numpy as np
        return np.array(self.pil())

    def to_db(self, session, **kwargs) -> Artifact:
        """
        Convert the image to an artifact in the database.
        """
        datastore = db.get_or_create(session, Datastore, name="Local Filesystem")
        artifact_type = db.get_or_create(session, ArtifactType, name="Image")
        artifact = Artifact(
            name=self.name,
            datastore=datastore,
            artifact_type=artifact_type,
            artifact_path=self.uri
            **kwargs
        )
        session.add(artifact)
        session.commit()
        self._artifact = artifact
        return artifact
    
    def get_annotations(self) -> List[Annotation]:
        """
        Get the annotations for this image.
        """
        if self.artifact is None:
            raise ValueError("You must call to_db before calling get_annotations.")
        return self.artifact.annotations
    
    def annotate(self, label: str, xmin: float, ymin: float, xmax: float, ymax: float, confidence: float = None, *, session=None, **kwargs) -> Annotation:
        """
        Annotate the image with the object detection properties.
        """
        annotation = ObjectDetectionAnnotation(
            label=label,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            confidence=confidence,
            **kwargs
        )
        if session is not None:
            self.add_annotation(annotation, session=session)
            if self.artifact is None:
                self.to_db(session)
            annotation.to_db(self.artifact, session=session)
        return annotation

    def add_annotation(self, annotation: "ObjectDetectionAnnotation", session: object) -> None:
        """
        Add an annotation to the image artifact in the database.
        """
        if self.artifact is None:
            self.to_db(session)
        annotation.to_db(self.artifact, session=session)
    
    def display(self, *, ax=None, labels: bool = True, axis='off', annotations:bool=True, **kwargs):
        """
        Display the image along with the bounding boxes (annotations)
        """
        # from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        img_path = Path(self.uri)
        if not img_path.exists():
            raise FileNotFoundError(f"Could not find image at {img_path}")
        
        image_path = img_path.resolve()
        image = plt.imread(image_path)
        if ax is None:
            fig, ax = plt.subplots(1, 1, frameon=False)
        ax.imshow(image)
        # make the image fill the whole plot
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        # remove border around image
        if self.artifact is None:
            return ax
        if annotations:
            for i, rect in enumerate(self.rects):
                label = kwargs.pop('label', rect.label if labels and i==0 else None)
                rect.plot(ax=ax, label=label, **kwargs)
            if labels: 
                ax.legend()
        ax.axis(axis)
        plt.tight_layout()
        return ax
    
    @classmethod
    def from_db(cls, artifact: Artifact, **kwargs) -> "ImageForObjectDetection":
        """
        Create an ImageForObjectDetection from an artifact in the database.
        """
        # if artifact.artifact_type.name != "Image":
        #     raise ValueError("The artifact must be an image.")
        odimg = cls(
            uri=artifact.uri,
            name=artifact.name,
            **kwargs
            # artifact=artifact
        )
        odimg._artifact = artifact
        return odimg
    
    def to_xml(self, f:Optional[Union[IO, str]] = None, outer_label: str = "annotation"):
        """
        Convert the object to XML annotations.
        """
        from lxml.etree import Element, ElementTree

        root = Element(outer_label)
        filename_element = Element("filename")
        filename_element.text = self.uri
        root.append(filename_element)
        path_element = Element("path")
        path_element.text = str(self.uri)
        root.append(path_element)
        size_element = Element("size")
        width_element = Element("width")
        width_element.text = str(self.width)
        height_element = Element("height")
        height_element.text = str(self.height)
        size_element.append(width_element)
        size_element.append(height_element)
        root.append(size_element)
    
        if self.artifact is not None:
            for anno in self.artifact.annotations:
                anno_element = Element("object")
                properties_df = db.utils.to_df(anno.properties)[["name", "numeric_value", "text_value"]]
                for _, row in properties_df.iterrows():
                    if row["name"] == "label":
                        label_element = Element(row["name"])
                        label_element.text = row["text_value"]
                        anno_element.append(label_element)
                    elif row["name"] in ["xmin", "ymin", "xmax", "ymax"]:
                        bbox_element = Element(row["name"])
                        bbox_element.text = str(row["numeric_value"])
                        anno_element.append(bbox_element)
                root.append(anno_element)
        tree = ElementTree(root)
        if f is None:
            return tree
        elif isinstance(f, str):
            tree.write(f)
        else:
            tree.write(f)
        
class ObjectDetectionAnnotation(BaseModel):
    label : str = Field(..., description="The label of the object detected.")
    xmin : float = Field(..., description="The x-coordinate of the bottom-left corner of the bounding box.")
    ymin : float = Field(..., description="The y-coordinate of the bottom-left corner of the bounding box.")
    xmax : float = Field(..., description="The x-coordinate of the top-right corner of the bounding box.")
    ymax : float = Field(..., description="The y-coordinate of the top-right corner of the bounding box.")
    confidence : float = Field(None, description="The confidence level of the object detection.")
    width : Optional[float] = Field(None, description="The width of the bounding box.", gt=0)
    height : Optional[float] = Field(None, description="The height of the bounding box.", gt=0)
    name: Optional[str] = Field(None, description="The name of the annotation.")

    def to_db(self, artifact: Artifact, annotator: Annotator, session: object = None) -> Annotation:
        """
        Convert the object detection properties to an annotation property.
        """
        annotation = Annotation(
            name=self.name,
            annotator=annotator,
            artifact=artifact
        )
        if session is not None:
            session.add(annotation)
        properties = []
        for prop_name, field in self.__fields__.items():
            if prop_name == "name":
                continue
            property_value = getattr(self, prop_name)
            if property_value is None:
                continue
            numeric_value, text_value =  None, None
            if isinstance(property_value, numbers.Number):
                numeric_value = float(property_value)
            else:
                text_value = str(property_value)
            properties.append(
                AnnotationProperty(
                    name=prop_name,
                    numeric_value=numeric_value,
                    text_value=text_value,
                    annotation=annotation
                )
            )
            if session is not None:
                session.add(properties[-1])
        if session is not None:
            session.commit()
        else:
            annotation.properties = properties
        return annotation
    
    def to_xml(self, f:Optional[Union[IO, str]] = None, outer_label:str="object"):
        """
        Convert the object detection properties to an XML annotation.
        """
        xml = f"<{outer_label}>\n"
        for prop_name, value in self.dict().items():
            if value is None or prop_name == "name":
                continue
            xml += f"  <{prop_name}>{value}</{prop_name}>\n"
        xml += f"</{outer_label}>"
        if f is not None:
            if isinstance(f, str):
                with open(f, "w") as f:
                    f.write(xml)
            else:
                f.write(xml)
        return xml
    
    @classmethod
    def from_db(cls, annotation: Annotation) -> "ObjectDetectionAnnotation":
        properties = annotation.properties
        property_dict = {
            prop.name: prop.numeric_value or prop.text_value
            for prop in properties
        }
        return cls(**property_dict)
