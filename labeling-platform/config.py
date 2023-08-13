import os, dotenv
import logging
from pathlib import Path

from sqlalchemy import create_engine
import streamlit as st

# logging
logging.basicConfig(level=logging.INFO)

# environment
dotenv.load_dotenv()

# streamlit
st.set_option("deprecation.showfileUploaderEncoding", False)

# database
db_conn_str = os.environ.get("DB_CONNECTION_STRING", None)
db_engine = create_engine(db_conn_str)

# directory where annotations are stored
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", './'))
# directory where to run the streamlit app
WORKING_DIR = Path(os.environ.get("WORKING_DIR", './'))
# labels per annotation project
ANNOTATION_TASKS = {
    "Sputum Detection": [
        "Sputum",
    ],
    "Covid Detection": [
        "Covid",
    ],
}
# models for autolabeling and their URIs
MODEL_INFO = {
    "rCNN Sputum Detector v1": 
        {
            "name": os.environ.get("MODEL_NAME", "rcnn_MNasNet_2"),
            "model": os.environ.get("MODEL_URI","models:/sputum_detector/13"),
            "uri": os.environ.get("MODEL_INVOCATIONS_URI","http://127.0.0.1:8888/predict"),
        }
}