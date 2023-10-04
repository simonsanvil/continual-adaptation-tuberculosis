FROM python:3.8-slim-buster

# A Dockerfile to initialize the labeling platform with streamlit

# Set the working directory
WORKDIR /usr/src/app

# instal gcc and other dependencies
RUN apt-get clean && apt-get update && apt-get install -y gcc g++ build-essential

ADD streamlit_img_label/ /usr/src/app/streamlit-img-label/
ADD app.py /usr/src/app/
ADD *annotation*.py /usr/src/app/
ADD config.py /usr/src/app/
ADD requirements.txt /usr/src/app/

RUN mkdir -p /usr/src/app/data/

# install the required packages
RUN pip install -r /usr/src/app/requirements.txt
RUN pip install -e /usr/src/app/streamlit-img-label/

ENV STREAMLIT_PORT=${STREAMLIT_PORT:-8502}
ENV STREAMLIT_SERVER_HOST=${STREAMLIT_SERVER_HOST:-0.0.0.0}
ENV ANNOTATIONS_DIR=${ANNOTATIONS_DIR:-/usr/src/app/data/annotations}
ENV IMAGES_DIR=${IMAGES_DIR:-/usr/src/app/data/images}

EXPOSE ${STREAMLIT_PORT}
WORKDIR /usr/src/app/

# Run the streamlit server
# CMD streamlit run app.py --serverC.port $STREAMLIT_PORT --server.headless true --server.enableCORS false --server.address $STREAMLIT_SERVER_HOST