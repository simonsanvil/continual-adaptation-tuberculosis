FROM mlflow:latest

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
ENV SERVED_MODEL_URI=${SERVED_MODEL_URI:-models:/}
ENV SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
ENV SERVED_MODEL_VERSION=${SERVED_MODEL_VERSION}
ENV SERVED_MODEL_PORT=${SERVED_MODEL_PORT:-8000}
ENV PYENV_ROOT=${PYENV_ROOT:-/root/.pyenv}

# Set the working directory
WORKDIR /usr/src/app

ADD requirements.txt /usr/src/app/requirements.txt
# Install dependencies
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ADD sputum_detection_model.py /usr/src/app/sputum_detection_model.py
ADD inference.py /usr/src/app/inference.py
ADD load.py /usr/src/app/load.py

# Run the mlflow model server
# ENTRYPOINT ["python", "inference.py", "--port", ${SERVED_MODEL_PORT}, "--host", "0.0.0.0"]