from .models import (
    Project,
    Annotation,
    Artifact,
    Datastore,
    AnnotationProperty,
    Annotator,
    ArtifactType,
    ProjectAnnotator,
    ArtifactTag,
    SessionState,
)

from .session import get_session
from .utils import __all__ as __utils_all__

__all__ = __utils_all__ + ["get_session"] + ["get_or_create"] + [
    "Project",
    "Annotation",
    "Artifact",
    "Datastore",
    "AnnotationProperty",
    "Annotator",
    "ArtifactType",
    "ProjectAnnotator",
    "ArtifactTag",
    "SessionState",
]

def get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        session.commit()
        return instance