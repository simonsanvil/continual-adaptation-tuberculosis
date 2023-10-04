FROM python:3.8-slim-buster

ENV SERVED_MODEL_PORT=${SERVED_MODEL_PORT:-8000}

COPY . /usr/src/app/

# Set the working directory
WORKDIR /usr/src/app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r /usr/src/app/requirements.txt

# # Run the FastAPI server
# ENTRYPOINT ["python", "/usr/src/app/app.py", "--port", ${SERVED_MODEL_PORT}]