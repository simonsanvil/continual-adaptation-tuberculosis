"""
abstract base classes of the database orc models
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = [
    "Artifact",
    "Annotation",
    "Project",
    "Datastore",
    "AnnotationProperty",
    "Annotator",
    "ArtifactType",
    "ProjectAnnotator",
    "ArtifactTag"
]

class Artifact(ABC):
    pass

class ArtifactTag(ABC):
    pass

class Annotation(ABC):
    pass

class Project(ABC):
    pass

class Datastore(ABC):
    pass

class AnnotationProperty(ABC):
    pass

class Annotator(ABC):
    pass

class ArtifactType(ABC):
    pass

class ProjectAnnotator(ABC):
    pass

class SessionState(ABC):
    pass