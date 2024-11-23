FROM docker.io/tfeltin/python3-ort-opencv:jetson

ENV PROVIDERS="['CUDAExecutionProvider', 'CPUExecutionProvider']"

COPY wedge-worker/requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY wedge-worker .