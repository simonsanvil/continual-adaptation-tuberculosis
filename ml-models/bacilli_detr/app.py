import os, logging, sys
import numpy as np
from typing import List, Dict

import fastapi, typer
import torch
import torchvision.transforms as T
from fastapi import Request
from PIL import Image

sys.path.append('/Users/simon/Documents/Projects/TFM')

from bacili_detection.detr.models import detr_inference

app = fastapi.FastAPI()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
formatter = logging.Formatter('%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s')
logger.handlers[0].setFormatter(formatter)
# file_handler = logging.FileHandler("app.log")
# logger.addHandler(file_handler)

DETR_MODEL_PATH = os.environ.get("DETR_MODEL_PATH", "bacili_detection/detr/outputs/checkpoint.pth")
checkpoint = torch.load(DETR_MODEL_PATH, map_location='cpu')
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False, num_classes=2)
model.load_state_dict(checkpoint['model'])

infer_transform = T.Compose([  # These are necessary for inference with DETR
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.post("/predict")
async def predict(request: Request) -> Dict:
    """
    Predict the bounding boxes of the image.
    """
    data = await request.json()
    image = np.array(data["data"])
    logger.info(f"received data of type: {type(image)}")
    im = Image.fromarray(image.astype(np.uint8))
    logger.info(f"Received image with shape: {image.shape}")
    try:
        bboxes, confidences = detr_inference([im], model, infer_transform, labels=True)
        bboxes = bboxes[0]
        confidences = confidences [0][:,1]
    except Exception as e:
        logger.error(f"Error while predicting: {e}")
        raise fastapi.HTTPException(status_code=500, detail="Internal server error")
    return {"bboxes": bboxes.tolist(), "confidences": confidences.tolist(), "label": "TBbacilli"}

def main(host:str="0.0.0.0", port:int=8000):
    import uvicorn, sys

    logger.info(f"Starting server at {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    typer.run(main)