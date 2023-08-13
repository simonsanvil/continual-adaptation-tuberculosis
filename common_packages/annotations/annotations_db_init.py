from typing import List, Optional
from pydantic import BaseModel, Field
import os, numbers
import sqlalchemy

from annotations import db
from annotations.db.models import (
    AnnotationProperty, Annotation, Artifact, 
    Project, Datastore, Artifact, ArtifactType, 
    Annotation, AnnotationProperty, Annotator, ProjectAnnotator
)
from annotations.object_detection.object_detection import ImageForObjectDetection, ObjectDetectionAnnotation
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# create the database tables
from annotations.db.models import mapper_registry

# Create the database engine and session
# conn_str = 'sqlite:///annotations.db'
# conn_str = os.environ.get("DB_CONN_STR", conn_str)
conn_str = 'sqlite:///annotations.db'
engine = create_engine(conn_str, echo=False)
Session = sessionmaker(bind=engine)
session = Session()

# create the database tables
mapper_registry.metadata.create_all(engine)

# Populate the database with some data
project = Project(
    name="Sputum Detection",
    description="Detect sputum in images."
)
session.add(project)

datastore = Datastore(
    name="Local Filesystem",
    uri="file:///",
    description="The local filesystem of a machine running the project.",
)

session.add(datastore)

artifact_types = [
    ArtifactType(name="Image"),
    ArtifactType(name="Video"),
    ArtifactType(name="Audio"),
    ArtifactType(name="Text"),
    ArtifactType(name="Model"),
    ArtifactType(name="CSV"),
    ArtifactType(name="JSON"),
    ArtifactType(name="XML"),
    ArtifactType(name="YAML"),
    ArtifactType(name="Other")
]

session.add_all(artifact_types)

session.commit()

annotator = Annotator(
    name="Simon Sanchez Viloria",
    email="simsanch@inf.uc3m.es",
    automatic=False,
    description="A human annotator."
)

session.add(annotator)

project_annotator = ProjectAnnotator(
    project=project,
    annotator=annotator
)

session.add(project_annotator)

session.commit()

# make an automatic annotator for sputum detection
auto_annotator = Annotator(
    name="rCNN Sputum Detector v1",
    email=None,
    description="An automatic annotator for sputum detection (rcnn MNasNet 2).",
    automatic=True,
)
auto_annotator = Annotator(
    name="Simon Sanchez Viloria",
    email='simsanch@inf.uc3m.es',
    description="An automatic annotator for sputum detection (rcnn MNasNet 2).",
    automatic=True,
)
session.add(auto_annotator)
session.commit()
session.add(ProjectAnnotator(project=project, annotator=auto_annotator))
session.commit()

# Create an artifact
artifact = Artifact(
    name="tuberculosis-phone-0016",
    description="An image of sputum sourced from https://www.kaggle.com/datasets/saife245/tuberculosis-image-datasets",
    uri="data/images/val-tuberculosis-phone-0016.jpg",
    datastore=datastore,
    project=project,
    artifact_type=next(filter(lambda x: x.name == "Image", artifact_types))
)
session.add(artifact)
session.commit()