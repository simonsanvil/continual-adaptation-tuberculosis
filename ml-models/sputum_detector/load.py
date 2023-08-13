import os
import yaml, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
file_handler = logging.FileHandler("app.log")
logger.addHandler(file_handler)

def load_model(config_path:str="config.yaml"):
    import dotenv
    import mlflow
    from ml_collections import config_dict
    from mlflow.keras import load_model as load_keras_model
    from sputum_detection_model import SputumDetectionModel

    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    config = config_dict.ConfigDict(cfg)

    # dotenv.load_dotenv(config.environment.dotenv_path)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config.environment.variables.MLFLOW_TRACKING_URI)
    logger.info(f"Attempting to load model from {config.model.uri}. This may take a while...")
    mlflow.set_tracking_uri(tracking_uri)

    keras_model = load_keras_model(config.model.uri)
    model = SputumDetectionModel(keras_model, **config.model.params)
    logger.info(f"Successfully loaded model from {config_path}")
    return model

