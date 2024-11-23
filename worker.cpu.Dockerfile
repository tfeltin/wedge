FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir onnx onnxruntime opencv-python-headless paho-mqtt networkx

COPY wedge-worker .
