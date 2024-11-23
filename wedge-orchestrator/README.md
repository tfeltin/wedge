
# Wedge Orchestrator

The orchestrator is the central management and monitoring entity of the Wedge framework, which allows the interaction with the different workers, to pull runtime information and push new placements dynamically. The orchestrator is also the entry point to use the Wedge service through an API.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/31653233/389213073-d3bd2d22-cd4b-4694-af3b-afd2ffe9d55d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzIzODExMTQsIm5iZiI6MTczMjM4MDgxNCwicGF0aCI6Ii8zMTY1MzIzMy8zODkyMTMwNzMtZDNiZDJkMjItY2Q0Yi00Njk0LWFmM2ItYWZkMmZmZTlkNTVkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTIzVDE2NTMzNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWNmNmRjNDAzYTExNGEzNzQ4ODJjMjE2ZDVhMWUyZTE2ODFkYThmNzNhZTUzZDI0MDdkZTg4MWMwNjgyNjAwYzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.upuFa0ZMg6r7UsqXUllx3IKQLT_nNQKqfqPf3QxuboQ" width="500" />
</p>

## Usage

To launch the orchestrator, after having set the right environment variables, run
```
python main.py
```

## Configuration Variables

| Environment Variable | Purpose |
| - | - |
| WORKER_ADDRESSES | *(Required)* Lists the host addresses of all the workers as a list of strings, *e.g.,* ```['10.0.0.5','10.0.0.6','10.0.0.7']```
| DNN_EXTRACT_ITERATIONS | *(Optional, default to ```5```)* Sets the number of iterations to extract the DNN profile |
| MAX_SPLIT | *(Optional, default to ```5```)* Sets the maximum number of partitions for the optimal placement (early stopping) |
| LOGGING | *(Optional, default to False)* Show real computation/transmission times compared to predicted values by the orchestrator |
| API_PORT | *(Optional, default to ```6968```)* Sets the API port used by applications to reach the Wedge service |
| SYNC_PORT | *(Optional, default to ```6869```)* Sets the Sync port used by the workers to listen for updates by the orchestrator |
| REGISTRATION_PORT | *(Optional, default to ```6967```)* Sets the registration port used for workers to communicate with the orchestrator |

## Prerequisites
The wedge orchestrator requires at least a quadcore machine.  
When using VM set cpus = 4 