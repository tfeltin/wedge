import os
import json
from utils.sync import SyncAgent
from utils.api import APIServer
from utils.register import RegistrationAgent
from utils.log import Logger
from model.load import download_model
from model.profiler import extract_onnx_dnnconf
from placement.placement import bnb, conf_to_matrix, max_TcTt, plPartsToPl


def main(config):
    sync_agent = SyncAgent(config)
    api_server = APIServer(config)
    model_path = None
    model_url = None
    io = None
    netconf = None
    dnnconf = None
    placement = None
    logger = None

    # Agent for workers to register
    registration_agent = RegistrationAgent(config, api_server)

    print("* Wedge 2.0 Orchestrator is ready. ")

    while True:
        # Wait for update from API server
        update = api_server.wait_for_update()

        # API server received new registration request
        if update['action'] == 'new_worker':
            if update['payload']['worker_address'] in config['worker_addresses']:
                worker_number = config['worker_addresses'].index(update['payload']['worker_address'])
                sync_agent.init_worker(worker_number)
            else:
                worker_number = config['N_workers']
                config['N_workers'] += 1
                config['worker_addresses'].append(update['payload']['worker_address'])
                sync_agent.init_new_worker(worker_number, update['payload']['worker_address'])
            sync_agent.check_bandwidth_state(worker_number)
            sync_agent.claim_worker(worker_number)
            sync_agent.push_worker_addresses_all()
            print("* Claimed worker %s (%s)" % (worker_number,  update['payload']['worker_address']))

        # API server received new input/output source
        if update['action'] == 'new_io':
            io = update['payload']
            print("\n* Pushing I/0 to workers")
            sync_agent.push_io_to_all(update['payload'])
        
            if io['source'] is None or io['dest'] is None:
                io = None

        # API server received new model
        if update['action'] == 'new_model':
            model_url = update['payload']
            print("\n* Pushing model %s to workers" % model_url)
            sync_agent.push_model_to_all(model_url)

            if model_url is None:
                model_path = None
                dnnconf = None
                continue

            # Extract DNN configuration
            print("\n* Extracting DNN configuration")
            model_path = download_model(model_url)
            dnnconf = extract_onnx_dnnconf(model_path, N_iter=config['dnn_iterations'])

            print("  DNN configuration extracted")
            if config['logging']:
                print('=' * 33)
                print("EXTRACTED DNN CONFIGURATION:")
                print('-' * 33)
                print(json.dumps(dnnconf, indent=2))
                print('=' * 33)

            # Wait for workers to finish loading model + benchmarking node
            sync_agent.wait_for_model_load()
        
        # Wedge orchestrator received I/O + model and can start inference
        if (update['action'] == 'new_io' or update['action'] == 'new_model') and model_path is not None and io is not None:

            # Trigger bandwidth computations and extract network configuration
            print("\n* Extracting network configuration")
            netconf = sync_agent.get_netconf(dnnconf, config)
    
            if config['logging']:
                print('=' * 33)
                print("EXTRACTED NETWORK CONFIGURATION:")
                print('-' * 33)
                print(json.dumps(netconf, indent=2))
                print('=' * 33)

            # Compute optimal placement (Branch & Bound algorithm)
            print("\n* Computing optimal placement")
            placement, best_time = bnb(dnnconf, netconf, max_split=config['max_split'], parallel=config['parallel'])

            if config['logging']:
                print('=' * 33)
                print(" Optimal placement:\t%s" % placement)
                print(" Optimal cadence:\t%s FPS" % round(1/best_time, 2))
                print('=' * 33)
                print()
            
            # Push optimal placement to workers
            print("  Pushing placement to workers")
            sync_agent.update_placement(placement)
            sync_agent.push_placement_to_all()

            if config['logging']:
                logger = Logger(placement, dnnconf, netconf, sync_agent, config)
                logger.start()
        
        # API server received command to recompute optimal placement 
        if update['action'] == 'compute_placement':

            print("\n* Recomputing optimal placement")
            netconf = sync_agent.get_netconf(dnnconf, config)
            print('=' * 33)
            print("EXTRACTED NETWORK CONFIGURATION:")
            print('-' * 33)
            print(json.dumps(netconf, indent=2))
            print('=' * 33)
            
            c, K, S, B = conf_to_matrix(dnnconf, netconf)
            best_time = max_TcTt(plPartsToPl(placement), c, K, S, B)

            new_placement, new_best_time = bnb(dnnconf, netconf, max_split=config['max_split'], parallel=config['parallel'])

            if config['logging']:
                logger.stop()
                print('=' * 33)
                if new_placement != placement and new_best_time < best_time:
                    print("  New placement:\t%s" % new_placement)
                    print("  New estimated cadence:\t%s FPS (current: %s)" % (round(1/new_best_time, 2), round(1/best_time, 2)))
                    placement, best_time = new_placement, new_best_time
                    print("  Pushing updated placement to workers")
                    sync_agent.update_placement(placement)
                    sync_agent.push_placement_to_all()
                elif new_placement == placement:
                    print("  Current placement is optimal")
                    print("  New estimated cadence:\t%s FPS" % round(1/new_best_time, 2))
                elif new_placement != placement:
                    print("  New best placement is worst than current one")
                    print("  New estimated cadence:\t%s FPS (current: %s)" % (round(1/new_best_time, 2), round(1/best_time, 2)))
                print('=' * 33)

                if config['logging']:
                    logger = Logger(placement, dnnconf, netconf, sync_agent, config)
                    logger.start()


if __name__ == '__main__':
    print('-' * 33)
    print("|     WEDGE 2.0 ORCHESTRATOR\t|")
    print('-' * 33)
    config = {}

    # Worker information
    config['worker_addresses'] = []
    config['N_workers'] = 0

    # Number of iterations in DNN profiling
    config['dnn_iterations'] = os.environ.get('DNN_EXTRACT_ITERATIONS')
    if config['dnn_iterations'] is None:
        config['dnn_iterations'] = 5
    else:
        config['dnn_iterations'] = int(config['dnn_iterations'])

    # Max number of partitions in optimal placement (early stopping)
    config['max_split'] = os.environ.get('MAX_SPLIT')
    if config['max_split'] is None:
        config['max_split'] = 5
    else:
        config['max_split'] = int(config['max_split'])
    
    # Allowing parallel execution of partitions on same node
    if os.environ.get('PARALLEL') is None or os.environ.get('PARALLEL') not in ('True', 'False'):
        config['parallel'] = True
    else:
        config['parallel'] = eval(os.environ.get('PARALLEL'))
    
    # Logging
    if os.environ.get('LOGGING') is None or os.environ.get('LOGGING') not in ('True', 'False'):
        config['logging'] = False
    else:
        config['logging'] = eval(os.environ.get('LOGGING'))
    
    # Sync Port
    config['sync_port'] = os.environ.get('SYNC_PORT')
    if config['sync_port'] is None:
        config['sync_port'] = 6869
    else:
        config['sync_port'] = int(config['sync_port'])
    
    # API Port
    config['api_port'] = os.environ.get('API_PORT')
    if config['api_port'] is None:
        config['api_port'] = 6968
    else:
        config['api_port'] = int(config['api_port'])

    # Registration Port
    config['registration_port'] = os.environ.get('REGISTRATION_PORT')
    if config['registration_port'] is None:
        config['registration_port'] = 6967
    else:
        config['registration_port'] = int(config['registration_port'])
    
    main(config)
