import os
from datetime import datetime
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass, field
from functools import cached_property

from sqlalchemy import UniqueConstraint, inspect, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship

import pandas as pd
from sqlalchemy import inspect, func
from sqlalchemy.orm import relationship, synonym, registry, column_property
from sqlalchemy.sql import func, or_
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Float,
    Date,
    Table,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

mapper_registry = registry()

from ._base import Annotator
from ._base import *

__all__ = [
    "Project",
    "Datastore",
    "ArtifactType",
    "Artifact",
    "ArtifactTag",
    "Annotator",
    "Annotation",
    "AnnotationProperty",
    "ProjectAnnotator",
    "SessionState",
]

@mapper_registry.mapped
@dataclass(eq=False)
class Project:
    """
    A project is a collection of artifacts that are annotated by a group of annotators.
    """
    __table__ = Table(
        "projects",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, unique=True),
        Column("description", String, default=""),
        Column("created_at", DateTime, server_default=func.now()),
    )

    id: int = field(init=False)
    name: str = None
    description: str = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    artifacts: List["Artifact"] = field(default_factory=list)
    annotators: List["Annotator"] = field(default_factory=list)

    __mapper_args__ = {
        "properties": {
            "annotators": relationship(
                "Annotator", secondary="project_annotators", back_populates="projects"),
        }
    }

    def __repr__(self):
        return f"<Project {self.name}>"

@mapper_registry.mapped
@dataclass(eq=False)
class Datastore:
    """
    A datastore is the source of an artifact. It can be a local directory, a remote
    directory, a database, etc.
    """
    __table__ = Table(
        "datastores",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, unique=True),
        Column("description", String, server_default="", default=""),
        Column("created_at", DateTime, server_default=func.now(), nullable=False),
    )

    id: int = field(init=False)
    name: str = None
    description: str = field(default="")
    uri: str = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    __mapper_args__ = {
        "properties": {
        }
    }

@mapper_registry.mapped
@dataclass(eq=False)
class ArtifactType:

    """
    An artifact type is the type of artifact. It can be an image, a video, a text file,
    a table, a csv file, etc.
    """
    __table__ = Table(
        "artifact_types",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, unique=True),
        Column("created_at", DateTime, server_default=func.now()),
    )

    id: int = field(init=False)
    name: str = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    __mapper_args__ = {
        "properties": {
        }
    }


@mapper_registry.mapped
@dataclass(eq=False)
class Artifact:
    """
    An artifact is a file that is stored in a datastore that is associated with a project.
    Artifacts are often the object of the annotation process. That is, the annotator
    will annotate the artifact.
    
    An artifact can be an image, a video, a text file, a table, a csv file, etc.
    """
    __table__ = Table(
        "artifacts",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("project_id", Integer, ForeignKey("projects.id")),
        Column("datastore_id", Integer, ForeignKey("datastores.id")),
        Column("artifact_type_id", Integer, ForeignKey("artifact_types.id")),                       
        Column("name", String),
        Column("uri", String),
        Column("created_at", DateTime, server_default=func.now()),
    )

    id: int = field(init=False)
    name: str = None
    description: str = None
    uri: str = None
    project: Project = None
    datastore: Datastore = None
    artifact_type: ArtifactType = None
    created_at: datetime = None
    annotations: List["Annotation"] = field(default_factory=list)
    tags: List["ArtifactTag"] = field(default_factory=list)

    __mapper_args__ = {
        "properties": {
            "project": relationship("Project", backref="artifacts"),
            "datastore": relationship("Datastore"),
            "artifact_type": relationship("ArtifactType")
        }
    }

    def __repr__(self):
        path = self.uri
        return f"Artifact(name={self.name}, path={path})"
    
    @property
    def full_path(self):
        return os.path.join(self.datastore.uri or "", self.uri)

@mapper_registry.mapped
@dataclass(eq=False)
class ArtifactTag:
    """
    An artifact label is a label that is associated with an artifact.
    """
    __table__ = Table(
        "artifact_tags",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("artifact_id", Integer, ForeignKey("artifacts.id")),
        Column("tag", String),
        Column("created_at", DateTime, server_default=func.now()),
    )

    id: int = field(init=False)
    artifact_id: int
    tag: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    # artifact: Artifact = None

    __mapper_args__ = {
        "properties": {
            "artifact": relationship("Artifact", backref="tags"),
        }
    }


@mapper_registry.mapped
@dataclass(eq=False)
class Annotator:
    """
    An annotator is a person or model that creates annotations.

    the type field "automatic" is used to distinguish between human annotators and
    automatic annotators (e.g. a ML model or a rule-based system that creates automatic annotations).
    """
    __table__ = Table(
        "annotators",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, unique=True),
        Column("email", String, unique=True),
        Column("description", String, default=""),
        Column("automatic", Boolean, default=False),
        Column("created_at", DateTime, server_default=func.now()),
    )

    id: int = field(init=False)
    name: str
    email: str = None
    description: str = None
    automatic: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    projects: List[Project] = field(default_factory=list)
    annotations: List["Annotation"] = field(default_factory=list)

    __mapper_args__ = {
        "properties": {
            "projects": relationship(
                "Project", secondary="project_annotators", back_populates="annotators"),
        }
    }

    def __repr__(self):
        return f"Annotator(name={self.name}, email={self.email})"

@mapper_registry.mapped
@dataclass(eq=False)
class ProjectAnnotator:
    """
    A table that links projects to annotators in a many-to-many relationship.
    i.e: One annotator can be associated with multiple projects, projects can have several annotators.
    """
    __table__ = Table(
        "project_annotators",
        mapper_registry.metadata,
        Column("project_id", Integer, ForeignKey("projects.id"), primary_key=True),
        Column("annotator_id", Integer, ForeignKey("annotators.id"), primary_key=True),
    )
    project_id: int = field(init=False)
    annotator_id: int = field(init=False)
    project: "Project" = None
    annotator: "Annotator" = None 

    def __post_init__(self):
        if self.project is not None:
            self.project_id = self.project.id
        if self.annotator is not None:
            self.annotator_id = self.annotator.id

@mapper_registry.mapped
@dataclass(eq=False)
class Annotation:
    """
    An annotation is the result of one annotator labeling an artifact. An artifact can have
    multiple annotations. Each annotation will be associated with an annotator.
    """
    __table__ = Table(
        "annotations",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("artifact_id", Integer, ForeignKey("artifacts.id")),
        Column("annotator_id", Integer, ForeignKey("annotators.id")),
        Column("name", String),
        Column("description", String),
        Column("created_at", DateTime, server_default=func.now()),
    )

    id: int = field(init=False)
    artifact: Artifact
    annotator: Annotator
    name: str = None
    description: str = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    properties: List['AnnotationProperty'] = field(default_factory=list)

    __mapper_args__ = {
        "properties": {
            "artifact": relationship('Artifact', backref="annotations"),
            "properties": relationship('AnnotationProperty', backref="annotation"),
            "annotator": relationship('Annotator', backref="annotations"),
        }
    }

    def get_property(self, name):
        return next((p for p in self.properties if p.name == name), None)


@mapper_registry.mapped
@dataclass(eq=False)
class AnnotationProperty:
    """
    An annotation property is a property of the specific annotation. The properties are defined by 
    the annotator or the particular application. E.g: for image classification, the annotator can
    define the properties as the confidence level of the classification, for object detection, the
    properties may include the bounding box of the object, etc. Most annotations will have 
    the "label" property set to their respective label.
    """
    __table__ = Table(
        "annotation_property",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("annotation_id", Integer, ForeignKey("annotations.id")),
        Column("name", String),
        Column("numeric_value", Float),
        Column("text_value", String),
        Column("created_at", DateTime, server_default=func.now()),
    )

    id: int = field(init=False)
    name: str = None
    numeric_value: float = None
    text_value: str = None
    annotation: Annotation = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    __mapper_args__ = {
        "properties": {
            # "annotation": relationship("Annotation", back_populates="properties")
        }
    }

@mapper_registry.mapped
@dataclass(eq=False)
class SessionState:
    """ 
    Records the state of a session at a given point in time. 
    A session is an instance of a user interacting with the annotation/labeling tool.
    """

    id: int = field(init=False)
    session_id: str
    annotator_id: int
    project_id: int
    session_state: JSONB
    created_at: datetime = field(default_factory=datetime.utcnow)

    __table__ = Table(
        "session_states",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("session_id", String),
        Column("annotator_id", Integer, ForeignKey("annotators.id")),
        Column("project_id", Integer, ForeignKey("projects.id")),
        Column("session_state", JSONB),
        Column("created_at", DateTime, server_default=func.now()),
    )

    __mapper_args__ = {
        "properties": {
            "annotator": relationship("Annotator", backref="session_states"),
            "project": relationship("Project", backref="session_states"),
        }
    }




# @mapper_registry.mapped
# @dataclass(eq=False)
# class AnnotationEvent:
#     """ 
#     An annotation event is a record of an annotation being created, updated or deleted.
#     It is used to keep track of the history of annotations as they are manipulated in the
#     annotation tool in real-time.
#     """

#     id: int = field(init=False)
#     annotation_id: int = field(init=False)
