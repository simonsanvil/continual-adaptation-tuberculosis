import json
from pathlib import Path
from typing import List

import requests
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session,  joinedload

from annotations import db
from annotations.db.models import (
    Project, Annotation, Annotator, Artifact, ArtifactType,
    AnnotationProperty
)
from annotations.object_detection.rect import Rect, Rects


def load_annotations(project_name:str, session: object = None):
    if session is None:
        session = db.get_session()

    project = session.query(Project).filter(Project.name == project_name).first()
    if project is None:
        raise ValueError(f"Project {project_name} not found.")
    return project.annotations


def load_artifacts(project_name:str, session:object=None):
    if session is None:
        session = db.get_session()
    artifacts = session.query(Artifact)\
        .join(Project)\
            .where(Project.name == project_name)
    
    return artifacts.all()

def get_artifact_bbox_annotations(artifact_annotations:list):
    """
    Get the relevant properties for the given annotations.
    """
    bboxes = []
    for annotation in artifact_annotations:
        bbox_dict = dict(
            annotator_name=annotation.annotator.name, 
            annotator_email=annotation.annotator.email, 
            annotator_automatic=annotation.annotator.automatic,
            bbox={})
        for property in annotation.properties:
            if property.name in ["xmin", "ymin", "xmax", "ymax"]:
                bbox_dict['bbox'][property.name] = property.numeric_value
            elif property.name == "label":
                bbox_dict[property.name] = property.text_value
        bboxes.append(bbox_dict)
    return bboxes

def parse_rects_from_annotations(artifact_annotations:List[Annotation]) -> Rects:
    """
    Get the relevant bbox rect annotations for the given artifact annotations.
    """
    rects_list = []
    for annotation in artifact_annotations:
        rect_dict = dict()
        for property in annotation.properties:
            if property.name in ["xmin", "ymin", "xmax", "ymax"]:
                rect_dict[property.name] = property.numeric_value
            elif property.name == "label":
                rect_dict[property.name] = property.text_value
        if len(rect_dict) < 5:
            continue
        rect = Rect(
            left=rect_dict.get('xmin'),
            top=rect_dict.get('ymin'),
            width=rect_dict.get('xmax') - rect_dict.get('xmin'),
            height=rect_dict.get('ymax') - rect_dict.get('ymin'),
            meta=dict(
                automated=rect_dict.get('automated'),
                annotation_id = annotation.id,
                annotator_name=annotation.annotator.name, 
                annotator_email=annotation.annotator.email,
                annotator_automatic=annotation.annotator.automatic
            )
        )
        rects_list.append(rect)

    return Rects(rects_list)

# def get_annotated_artifacts


def get_auto_annotators(project: Project):
    """
    Get the annotators that are automatic.
    """
    return [a for a in project.annotators if a.automatic]


def load_annotations_df(project_name:str, session: object = None):
    annotations_result = query_annotations(project_name, session)
    df = (
        pd.DataFrame([r._asdict() for r in annotations_result])
        .set_index(["project_name", "artifact_name", "artifact_uri", "annotation_name", "property_name"])
        .unstack("property_name").dropna(axis=1, how="all")
        .pipe(set_df_cols, lambda x: x[0] if not x[1] else x[1])
        .reset_index()
        .groupby("artifact_uri")
        .apply(
            lambda x: {
                "project_name": x["project_name"].iloc[0],
                "artifact_name": x["artifact_name"].iloc[0],
                "annotations_count": x.shape[0],
                "annotations": x[["label", "xmin", "ymin", "xmax", "ymax"]].to_dict(orient="records"),
            }
        ).apply(pd.Series)
        .reset_index()
    )
    return df

def query_annotations(project_name: str, session: object = None):
    q = select(
        Project.name.label("project_name"),
        Artifact.name.label("artifact_name"),
        Artifact.uri.label("artifact_uri"),
        Annotation.name.label("annotation_name"),
        Annotator.name.label("annotator_name"),
        Annotator.email.label("annotator_email"),
        Annotator.automatic.label("annotator_automatic"),
        AnnotationProperty.name.label("property_name"),
        AnnotationProperty.text_value, AnnotationProperty.numeric_value
    ).join(
        Artifact, Artifact.project_id == Project.id
    ).join(
        Annotation, Annotation.artifact_id == Artifact.id
    ).join(
        AnnotationProperty, Annotation.id == AnnotationProperty.annotation_id
    ).where(
        Project.name == project_name,
        AnnotationProperty.name.in_(["xmin", "ymin", "xmax", "ymax", "label"])
    )

    result = session.execute(q).fetchall()
    return result
    
def set_df_cols(df, mapping: callable):
    df.columns = df.columns.map(mapping)
    return df
    

def save_rects_to_db(
    rects: List[dict],
    artifact: db.Artifact,
    annotator: db.Annotator,
    session: object = None, 
    **kwargs
) -> None:
    """
    Save the given rects as new annotations to the database.
    """
    for rect in rects:
        pass


def get_projects(
        name: str = None,
        annotator_email: str = None,
        eager_load: bool = False, 
        session: object = None
    ):
    """
    Get the available projects.
    """
    if session is None:
        session = db.get_session()
    
    q = session.query(Project)
    if name:
        q = q.filter(db.Project.name == name)
    if annotator_email:
        q = q.join(db.ProjectAnnotator).join(db.Annotator)\
            .filter(db.Annotator.email == annotator_email)
    if eager_load:
        q = q.options(
            joinedload(db.Project.artifacts)
            .joinedload(db.Artifact.annotations) 
            .joinedload(db.Annotation.properties)
        ).options(
            joinedload(db.Project.artifacts)
            .joinedload(db.Artifact.annotations)
            .joinedload(db.Annotation.annotator)
        ).options(
            joinedload(db.Project.artifacts)
            .joinedload(db.Artifact.tags)
        ).options(
            joinedload(db.Project.annotators)
        )
    return q.all()

def get_rects_from_model(serving_uri:str, image:np.ndarray, request_params:dict=None,  **rects_kwargs):
    """
    Annotation the image by requesting the bounding box predictions to a model
    that is deployed on a serving platform.

    Parameters
    ----------
    serving_uri : str
        The URI of the serving platform.
    image : np.ndarray
        The image to be annotated.
    **kwargs : dict
        Additional parameters to be passed to the serving platform.

    Returns
    -------
    predictions : np.ndarray
        The bounding box predictions.
    """
    request_params = request_params or {}
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"data": np.array(image).tolist()})
    resp = requests.post(serving_uri, data=data, headers=headers, params=request_params)
    resp.raise_for_status()
    resp_data = resp.json()
    if "error" in resp_data:
        raise Exception(resp_data["error"])
    else:
        predictions = np.array(resp_data["bboxes"])
    rects = []
    for xmin, ymin, xmax, ymax in predictions.tolist():
        rects.append(
            {
                "left": xmin,
                "top": ymin,
                "width": xmax - xmin,
                "height": ymax - ymin,
                "meta": rects_kwargs.copy(),
            }
        )
    return Rects(rects=[Rect(**rect) for rect in rects])