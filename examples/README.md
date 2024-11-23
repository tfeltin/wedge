# Wedge deployment examples

This folder contains examples to get started with Wedge.

## Deploy Wedge as a service

To deploy the Wedge service, use one of the two sample deployments in either of the ```docker-compose``` or ```kubernetes``` folders. The Wedge service must me composed of one Wedge worker per node to consider in the placement and one Wedge orchestrator to manage the workers.

Once the Wedge service is deployed, the orchestrator logs should be

```
---------------------------------
|       WEDGE ORCHESTRATOR      |
---------------------------------
* Claimed worker ...
```

And the workers should show

```
---------------------------------
|          WEDGE WORKER         |
---------------------------------
Sync Server -- Worker claimed as node #<N>
Wedge Worker -- Node claimed
```

## Use Wedge as a service

To start interacting with the Wedge service, use the example given in ```2_wedge_example.ipynb```. A comparison with the standard way of using the ONNX runtime can be seen in ```1_onnx_example.ipynb```.
The methods needed to interact with the service are included in the ```wedge.py``` file.
