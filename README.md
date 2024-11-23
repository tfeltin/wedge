# Wedge
Repositories for computing optimal placement and running partitioned DNNs.

## Overview
The Wedge framework includes two main components:
- The **Wedge workers** (```wedge-woker```), which include an ONNX runtime, synchronization mechanisms, physical network monitoring and I/O management with other workers and a potential orchestrator.
- The **Wedge orchestrator** (```wedge-orchestrator```) manages the deployed workers, extract configuration from the DNN, and compute optimal placements based on physical network information retrieved from the workers. The orchestrator also includes an API to interact with the Wedge service.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/31653233/389213075-46e2e8c2-29d0-4e40-b0fb-35dbe09df95f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzIzODExMTQsIm5iZiI6MTczMjM4MDgxNCwicGF0aCI6Ii8zMTY1MzIzMy8zODkyMTMwNzUtNDZlMmU4YzItMjlkMC00ZTQwLWIwZmItMzVkYmUwOWRmOTVmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTIzVDE2NTMzNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMzMDQ0ZGI3NGNjMzY1NjNhNTNlYzI4M2U4ZWE5ZDNiZGE5M2EzMmNhMzVhYTcxZDMzOWUyYTZjODE3MjI3MzImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.8OifnnHGB3exvKQDv_3rjOalqOXvadUiyMyin16u-6E" width="700" />
</p>

This repository provides both docker-compose and kubernetes support and configuration to run examples of the Wedge framewok.

## Example deployments

To run sample deployments on docker-compose or kubernetes, follow the steps described in the respective folders.
