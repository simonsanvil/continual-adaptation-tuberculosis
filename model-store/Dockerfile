FROM python:3.8-slim

# A Dockerfile to initialize the MLFlow model store

# Set the working directory
WORKDIR /usr/src/app

# install gcc and other dependencies
RUN apt-get update && apt-get install -y gcc g++ curl wget make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git

# Install pyenv
RUN curl https://pyenv.run | bash
# Add to PATH
ENV PATH="/root/.pyenv/bin:$PATH"

# Install MLFlow
RUN pip install "mlflow[extras]==2.2.1" "mlserver-mlflow==1.3.0.dev3"

# Create a directory for the model store
RUN mkdir -p usr/src/app/mlflow

# Set the model store as the working directory
WORKDIR /usr/src/app/mlflow

CMD ["bash"]

# Run the MLFlow server
# CMD mlflow server \
#     --host ${MLFLOW_SERVER_HOST} \
#     --port ${MLFLOW_PORT} \
#     --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} \
#     --backend-store-uri ${MLFLOW_BACKEND_STORE_URI}

# Build the image
# docker build -t mlflow-model-store .
# Run the image with the model store mounted
# docker run --mount /path/to/mlflow:/usr/src/app/mlflow -p 5000:5000 -e MLFLOW_MODEL_STORE=/usr/src/app/mlflow -e MLFLOW_PORT=5000 mlflow-model-store 