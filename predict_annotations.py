import os, requests, tqdm
from typing import Dict, List
from pathlib import Path

import numpy as np
from PIL import Image

from annotations.object_detection import ImageForObjectDetection
from annotations.object_detection.rect import Rect,Rects
from annotations import db

from dotenv import load_dotenv
load_dotenv()

def annotate_artifacts(
        artifacts: List[db.Artifact],
        serving_uri: str,
        annotator: db.Annotator,
        **request_params
    ) -> Dict[db.Artifact, Rects]:
    """
    Get all artifacts from the database.
    """
    rects_dict = dict()
    for i, artifact in enumerate(tqdm.tqdm(artifacts, desc="Annotating artifacts")):
        # Get the image from the artifact.
        image = ImageForObjectDetection.from_db(artifact)
        img_numpy = image.numpy()
        # Send the image to the serving uri.
        try:
            resp = requests.post(
                serving_uri,
                json={"data":img_numpy.tolist()},
                **request_params
            )
            pred = resp.json()
        except Exception as e:
            print(f"Error parsing response: {e}")
            continue
        rects = Rects([Rect.from_bbox(tuple(bbox), label='sputum') for bbox in pred['bboxes']])
        rects_dict[artifact] = rects
        if i % 50 == 0:
            print(f"Annotated {i+1} artifacts. Saving to database...")
            artifact_annotations = make_annotations(rects_dict, annotator=annotator)
            session.add_all(artifact_annotations)
            session.commit()
            print(f"Saved {len(artifact_annotations)} annotations.")
            rects_dict = dict()
    if len(rects_dict) > 0:
        print(f"Annotated {i+1} artifacts. Saving to database...")
        artifact_annotations = make_annotations(rects_dict, annotator=annotator)
        session.add_all(artifact_annotations)
        session.commit()
        print(f"Saved {len(artifact_annotations)} annotations.")

def make_annotations(
        rects_dict: Dict[db.Artifact, Rects],
        annotator: db.Annotator,
        session: object = None
    ):
    """
    Save the annotations to the database.
    """
    artifact_annotations = []
    for i, (artifact, rects) in enumerate(rects_dict.items()):
        for j, rect in enumerate(rects):
            annotation = db.Annotation(
                name=f"{Path(artifact.uri).stem}-rect-{i+1}",
                annotator=annotator,
                artifact=artifact,
                properties=[
                    db.AnnotationProperty(
                        name="xmin",
                        numeric_value=rect.left
                    ),
                    db.AnnotationProperty(
                        name="ymin",
                        numeric_value=rect.top
                    ),
                    db.AnnotationProperty(
                        name="xmax",
                        numeric_value=rect.right
                    ),
                    db.AnnotationProperty(
                        name="ymax",
                        numeric_value=rect.bottom
                    ),
                    db.AnnotationProperty(
                        name="label",
                        text_value=rect.label
                    )
                ]
            )
            artifact_annotations.append(annotation)
    return artifact_annotations

if __name__ == "__main__":
    headers = {"Content-Type": "application/json"}
    serving_uri = "http://127.0.0.1:8888/predict"
    with db.get_session(os.environ.get("DATABASE_URI")) as session:
        artifacts = session.query(db.models.Artifact).all()
        annotator = session.query(db.models.Annotator).filter(
            db.models.Annotator.name == "rCNN Sputum Detector v1"
        ).first()
        if annotator is None:
            raise ValueError("Annotator not found.")
        annotate_artifacts(
            artifacts,
            serving_uri=serving_uri,
            annotator=annotator,
            headers=headers
        )
