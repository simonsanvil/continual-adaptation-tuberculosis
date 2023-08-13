import logging

from mlengine import inference
import mlengine
import numpy as np

@inference.predict
def predict(model, data):
    """
    Predict the bounding boxes of the image.
    """
    image = data["data"]
    logging.info(f"received data of type: {type(image)}")
    image = np.array(image)
    logging.info(f"Received image with shape: {image.shape}")
    bboxes = model.predict_bbox(image)
    return {"bboxes": bboxes.tolist(), "label": "sputum"}

@inference.load
def load_model(config: dict):
    """
    Load the model from the config file.
    """
    import dotenv
    import mlflow
    from ml_collections import config_dict
    from mlflow.keras import load_model as load_keras_model
    from sputum_detection_model import SputumDetectionModel
    import os, logging

    logger = logging.getLogger(__name__)
    model_path = config.model_path
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config.environment.variables.MLFLOW_TRACKING_URI)
    logger.info(f"Attempting to load model from {config.model.uri}. This may take a while...")
    mlflow.set_tracking_uri(tracking_uri)

    keras_model = load_keras_model(config.model.uri)
    model = SputumDetectionModel(keras_model, **config.model.params)
    return model


# with mlengine.InferenceJob("sputum-object-detection") as j:
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     logger.addHandler(logging.StreamHandler())
#     file_handler = logging.FileHandler("app.log")
#     logger.addHandler(file_handler)

#     model = load_model(config_path=j.config_path)
#     j.run(model=model, predict=predict)

job = mlengine.InferenceJob(
    "sputum-object-detection",
    model_loading=load_model,
    predict_function=predict,
    config_path="config.yaml",
)

mlengine.register_job(job)