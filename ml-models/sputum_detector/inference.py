import os, logging
import numpy as np
from typing import List, Dict

import fastapi, typer
from fastapi import Request

from load import load_model

app = fastapi.FastAPI()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
formatter = logging.Formatter('%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s')
logger.handlers[0].setFormatter(formatter)
# file_handler = logging.FileHandler("app.log")
# logger.addHandler(file_handler)

model = load_model(config_path=os.environ.get("CONFIG_PATH","config.yaml"))

@app.post("/predict")
async def predict(request: Request) -> Dict:
    """
    Predict the bounding boxes of the image.
    """
    data = await request.json()
    image = data["data"]
    logger.info(f"received data of type: {type(image)}")
    image = np.array(image)
    logger.info(f"Received image with shape: {image.shape}")
    try:
        bboxes, confidences = model.predict(image)
    except Exception as e:
        logger.error(f"Error while predicting: {e}")
        raise fastapi.HTTPException(status_code=500, detail="Internal server error")
    return {"bboxes": bboxes.tolist(), "confidences": confidences.tolist(), "label": "sputum"}

def main(host:str="0.0.0.0", port:int=8000):
    import uvicorn

    logger.info(f"Starting server at {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    typer.run(main)
