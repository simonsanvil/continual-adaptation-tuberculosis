import mlflow.keras
import mlflow
import logging
import numpy as np
import cv2
from keras import load_model

class SputumDetectorPyfunc(mlflow.pyfunc.PythonModel):

    def __init__(self, **params):
        super().__init__()
        self.params = params
    
    def load_context(self, context):
        self.keras_model = load_model(context.artifacts["model"])

    def predict(self, context, img):
        img = np.array(img, dtype=np.uint8)
        out, chunks = self.detector.predict(img)
        return self.detector.get_bounding_boxes(img, out, chunks)
    
    @classmethod
    def log_keras(cls, model_name, keras_path, pip_requirements=None, **kwargs):     
        mlflow.pyfunc.log_model(
            model_name,
            python_model=cls(),
            artifacts = {"model": keras_path},
            code_path=[__file__],
            pip_requirements=pip_requirements,
        )
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("chunk_size", kwargs.pop("chunk_size", 80))
        mlflow.log_param("stride", kwargs.pop("stride", 40))
        mlflow.log_param("verbose", kwargs.pop("verbose", True))
        for key, value in kwargs.items():
            mlflow.log_param(key, value)