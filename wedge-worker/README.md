# Wedge Worker

The Wedge worker is the runtime to run partitioned models.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/31653233/389213076-bb4da2ea-a45e-4865-aaf2-3fbf04d48c18.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzIzODExMTQsIm5iZiI6MTczMjM4MDgxNCwicGF0aCI6Ii8zMTY1MzIzMy8zODkyMTMwNzYtYmI0ZGEyZWEtYTQ1ZS00ODY1LWFhZjItM2ZiZjA0ZDQ4YzE4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTIzVDE2NTMzNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc0NzZhYTg1ZTQzMGVhZjBmNTgyNjQyNGMzZDk0YjIxNWE5NzI4OWFhZDY5YTdiZGQxYWNkZTU4MTczY2FjYjYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.DuixOxwH5NaAqo3VkaxvL5HcO9xuAC5OZUsM0rj4aUY" width="700" />
</p>

## Running partitions

To run a partition on its own, or as a part of a deployment, after having set the right environment variables, run:
```
python main.py
```

## Configuration Variables

| Environment Variable  | Purpose |
| - | - |
| MODEL | *(Optional)* Sets the URL of the ONNX model used for partitioning, default is empty and can be provided by the orchestrator |                  
| SOURCE | *(Optional)* Sets the RTSP source URL to get input data from, default is empty and can be provided by the orchestrator |
| DEST | *(Optional)* Sets the MQTT broker URL to publish inference results to, default is empty and can be provided by the orchestrator |
| MQTT_TOPIC |  *(Optional, default to ```wedge_inference```)* Sets the MQTT topic used as output for inferences of the last partition |
| TCP_PORT | *(Optional, default to ```6868```)* Sets the TCP port used to communicate between workers |
| SYNC_PORT | *(Optional, default to ```6869```)* Sets the Sync port used to communicate with the orchestrator |
| REGISTRATION_PORT | *(Optional, default to ```6967```)* Sets the registration port used to communicate with the orchestrator |
