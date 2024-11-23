import os
from utils.sync import SyncAgent
from utils.metrics import benchmark_node
from utils.partition import start_partitions, stop_partitions
from model.load import download_model, load_model_info
from utils.register import register


def main(config):
    sync_server = SyncAgent(config)
    input_stream = None
    output_stream = None
    partitions = []

    # Announce oneself to the orchestrator
    register(config)

    while True:
        # Wait for instruction from orchestrator
        new_task = sync_server.wait_for_update()

        # Node claimed or worker addresses updated
        if new_task['action'] == 'CLAIM':
            config['worker_number'] = new_task['payload']['worker_number']
            config['worker_addresses'] = new_task['payload']['worker_addresses'].copy()
            print("Wedge Worker -- Node claimed as worker %s at %s" % (config['worker_number'], config['worker_addresses'][config['worker_number']]))
       
        # Worker received I/O
        elif new_task['action'] == 'PUSH_IO':
            if config['source'] != sync_server.state['source'] or config['dest'] != sync_server.state['dest']:
                config['source'] = sync_server.state['source']
                config['dest'] = sync_server.state['dest']

                if config['source'] is None or config['dest'] is None:
                    print("Wedge Worker -- Remove I/O (stopping partitions)")
                    # Stop partitions
                    stop_partitions(input_stream, output_stream, partitions, sync_server)
                else:
                    print("Wedge Worker -- New I/O -- Input : %s / Output : %s" % (config['source'], config['dest']))
                
        # Worker received model
        elif new_task['action'] == 'PUSH_MODEL':

            if sync_server.state['model'] is None:
                config['model'] = sync_server.state['model']
                print("Wedge Worker -- Remove model (stopping partitions)")
                sync_server.model_loaded = False
                # Stop partitions
                stop_partitions(input_stream, output_stream, partitions)

            else:
                model_path = download_model(sync_server.state['model'])
                
                print("Wedge Worker -- New model -- %s" % model_path)
                config['model'] = model_path
                sync_server.model_loaded = False
                
                # Benchmark model on node
                inference_time = benchmark_node(model_path, N_iter=config['benchmark_iter'])
                sync_server.update_time(0, inference_time)
                
                # Prepare model
                config['input_shape'], config['layers'] = load_model_info(model_path)
                sync_server.state['telemetry_placement'] = [[0, len(config['layers'])],[config['worker_number']]]
                sync_server.model_loaded = True
                print("Wedge Worker -- Model loaded and ready to compute")

        # Worker received placement
        elif new_task['action'] == 'PUSH_PLACEMENT':
            if config['model'] and sync_server.state['placement'][0][-1] != len(config['layers']):
                print("Wedge Worker -- Received placement with wrong number of layers (%s instead of %s)" % (sync_server.state['placement'][0][-1], len(config['layers'])))

            else:
                print("Wedge Worker -- New placement -- Placement %s : %s" % (sync_server.state['placement_id'], sync_server.state['placement']))
                # Update the placement
                config['placement'] = sync_server.state['placement']
                config['placement_id'] = sync_server.state['placement_id']

                # Stop partitions
                stop_partitions(input_stream, output_stream, partitions, sync_server)
                # Re-build and start new partitions
                input_stream, output_stream, partitions = start_partitions(sync_server, config)

        elif new_task['action'] == 'WORKER_ADDRESSES':
            config['worker_addresses'] = new_task['payload']['worker_addresses'].copy()
 

if __name__ == '__main__':
    print('-' * 33)
    print("|\t WEDGE 2.0 WORKER\t|")
    print('-' * 33)

    config = {'worker_number' : -1, 'worker_addresses' : ['localhost'], 
              'placement_id' : '000000', 'placement' : None}

    # Model
    config['model'] = os.environ.get('MODEL')

    # RTSP input
    config['source'] = os.environ.get('SOURCE')

    # MQTT output
    config['dest'] = os.environ.get('DEST')

    config['mqtt_topic'] = os.environ.get('MQTT_TOPIC')
    if config['mqtt_topic'] is None:
        config['mqtt_topic'] = "wedge_inference"
    
    # Inter node connectivity
    config['tcp_port'] = os.environ.get('TCP_PORT')
    if config['tcp_port'] is None:
        config['tcp_port'] = 6868
    
    # Sync server
    config['sync_port'] = os.environ.get('SYNC_PORT')
    if config['sync_port'] is None:
        config['sync_port'] = 6869
    
    # orchestrator
    config['orchestrator'] = os.environ.get('ORCHESTRATOR')
    assert config['orchestrator'] is not None, "Orchestrator address needs to be provided."
    if ":" in config['orchestrator']:
        config['registration_port'] = int(config['orchestrator'].split(":")[1])
        config['orchestrator'] = config['orchestrator'].split(":")[0]
    else:
        config['registration_port'] = 6967

    config['benchmark_iter'] = os.environ.get('BENCHMARK_ITERATIONS')
    if config['benchmark_iter'] is None:
        config['benchmark_iter'] = 20
    else:
        config['benchmark_iter'] = int(config['benchmark_iter'])

    main(config)
