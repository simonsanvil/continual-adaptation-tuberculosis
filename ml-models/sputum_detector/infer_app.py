import starlette
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import numpy as np
import logging
from load import load_model

from typing import Dict, List

def predict(model, data) -> Dict:
    """
    Predict the bounding boxes of the image.
    """
    logger = logging.getLogger(__name__)
    image = data["data"]
    image = np.array(image)
    logger.info(f"Received image with shape: {image.shape}")
    bboxes = model.predict_bbox(image)
    return {"bboxes": bboxes.tolist(), "label": "sputum"}

def load_model(config):
    import os
    import mlflow
    from mlflow.keras import load_model as load_keras_model
    from sputum_detection_model import SputumDetectionModel

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config.environment.variables.MLFLOW_TRACKING_URI)
    logger.info(f"Attempting to load model from {config.model.uri}. This may take a while...")
    mlflow.set_tracking_uri(tracking_uri)

    keras_model = load_keras_model(config.model.uri)
    model = SputumDetectionModel(keras_model, **config.model.params)
    return model


def make_predict_endpoint(config_path:str) -> Starlette:
    from ml_collections import config_dict
    import yaml, dotenv
    
    config = config_dict.ConfigDict(yaml.load(open(config_path, "r"), Loader=yaml.FullLoader))
    dotenv.load_dotenv(config.environment.dotenv_path)
    
    model = load_model(config)
    logging.info(f"Successfully loaded model from {config_path}")

    async def predict_wrapper(request) -> Dict:
        data = await request.json()
        result = predict(model, data)
        return JSONResponse(result)
    
    async def check_health(request):
        return JSONResponse({"status": "ok"})
    
    app = Starlette(debug=True, routes=[
        Route("/predict", predict_wrapper, methods=["POST"]),
        Route("/health", check_health, methods=["GET"])
    ])

    return app

def main(host:str="0.0.0.0", port:int=8000, config_path:str="config.yaml"):
    import uvicorn

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(f"Starting server at {host}:{port}")
    uvicorn.run(
        make_predict_endpoint(config_path=config_path), 
        host=host, port=port)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--config-path", default="config.yaml", type=str)
    args = parser.parse_args()
    main(**vars(args))