from pydantic import BaseModel, Field
from typing import List, Union

from annotations.object_detection.rect import Rects, Rect

class ObjectDetectionState(BaseModel):
    session_id: str = None
    project_id: int = None
    annotator_id:int = None
    current_artifact_id:int = None
    current_artifact_uri:str = None 
    selected_annotation_model:str = None
    previous_rects:List[dict] = Field(default_factory=dict)
    current_rects:List[dict] = Field(default_factory=dict)

    def __init__(self, **data):
        if "project" in data and "project_id" not in data:
            data["project_id"] = data["project"].id
        if "annotator" in data and "annotator_id" not in data:
            data["annotator_id"] = data["annotator"].id
        if data.get("current_artifact", data.get("artifact")) is not None:
            curr_artifact = data.pop("current_artifact", data.pop("artifact"))
            if "current_artifact_id" not in data:
                data["current_artifact_id"] = curr_artifact.id
            if "current_artifact_uri" not in data:
                data["current_artifact_uri"] = curr_artifact.uri

        if "previous_rects" in data:
            data['previous_rects'] = Rects(rects=[Rect.create(rect) for rect in data['previous_rects']])
        if "current_rects" in data:
            data['current_rects'] = Rects(rects=[Rect.create(rect) for rect in data['current_rects']])
        data['previous_rects'] = data['previous_rects'].todict()
        data['current_rects'] = data['current_rects'].todict()
        super().__init__(**data)