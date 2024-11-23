FROM docker.io/tfeltin/python3-ort-opencv:cpu

WORKDIR /usr/src/app

COPY wedge-orchestrator/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY wedge-orchestrator .

