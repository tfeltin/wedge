import queue
from threading import Thread
import socket
import pickle
import queue
from random import choice
from string import ascii_letters, digits
from time import sleep
from utils.utils import state_to_netconf


class SyncAgent:
    def __init__(self, config):
        self.port = int(config['sync_port'])
        self.N_workers = 0
        self.state = {'source': None, 'dest': None, 'placement_id': '000000', 'placement': None, 'worker_addresses' : [], 'model': None}
        self.bandwidth_state = {}
        self.model_state = {}
        self.update_queue = queue.Queue()
        self.sync_server_thread = Thread(target=self.run, daemon=True)
        self.sync_server_thread.start()

    def run(self):
        while True:
            update = self.update_queue.get()

            if update['action'] == 'PULL_STATE':
                serialized_message = pickle.dumps({'action': 'PULL_STATE'})
            elif update['action'] == 'PUSH_IO':
                serialized_message = pickle.dumps({'action': 'PUSH_IO', 'payload' : {'source': self.state['source'], 'dest': self.state['dest']}})
            elif update['action'] == 'PUSH_PLACEMENT':
                serialized_message = pickle.dumps({'action': 'PUSH_PLACEMENT', 'payload' : {'placement_id': self.state['placement_id'], 'placement': self.state['placement']}})
            elif update['action'] == 'PUSH_MODEL':
                serialized_message = pickle.dumps({'action': 'PUSH_MODEL', 'payload' : self.state['model']})
            elif update['action'] == 'BANDWIDTH_TEST':
                serialized_message = pickle.dumps({'action': 'BANDWIDTH_TEST', 'payload' : update['remote_host_list']})
            elif update['action'] == 'BANDWIDTH_STATE':
                serialized_message = pickle.dumps({'action': 'BANDWIDTH_STATE'})
            elif update['action'] == 'MODEL_STATE':
                serialized_message = pickle.dumps({'action': 'MODEL_STATE'})
            elif update['action'] == 'CLAIM':
                serialized_message = pickle.dumps({'action': 'CLAIM', 'payload': {'worker_number': update['worker_number'], 'worker_addresses': self.state['worker_addresses']}})
            elif update['action'] == 'PUSH_WORKER_ADDRESSES':
                serialized_message = pickle.dumps({'action': 'WORKER_ADDRESSES', 'payload': { 'worker_addresses': self.state['worker_addresses'], 'worker_number': update['worker_number']}})
          
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((self.state['worker_addresses'][int(update['worker_number'])], self.port))
                sock.sendall(serialized_message)
                if update['action'] == 'PULL_STATE':
                    pulled_state = sock.recv(1024)
                    pulled_state = pickle.loads(pulled_state)
                    self.state['node%s' % pulled_state['worker_number']] = pulled_state
                elif update['action'] == 'BANDWIDTH_STATE':
                    pulled_state = sock.recv(1024)
                    pulled_state = pickle.loads(pulled_state)
                    self.bandwidth_state['node%s' % pulled_state['worker_number']] = pulled_state['state']
                elif update['action'] == 'MODEL_STATE':
                    pulled_state = sock.recv(1024)
                    pulled_state = pickle.loads(pulled_state)
                    self.model_state['node%s' % pulled_state['worker_number']] = pulled_state['state']
                elif update['action'] == 'CLAIM':
                    pulled_state = sock.recv(1024)
                    pulled_state = pickle.loads(pulled_state)
                    self.bandwidth_state['node%s' % pulled_state['worker_number']] = True
                sock.close()
            except Exception as e:
                print("Sync Agent -- Connection to worker ", update['worker_number'], " addr ",
                            self.state['worker_addresses'][int(update['worker_number'])], " failed: ", e)
                sock.close()

    # Update I/O    
    def push_io(self, new_io, worker_number):
        if 'source' in new_io.keys():
            self.state['source'] = new_io['source']
        if 'dest' in new_io.keys():
            self.state['dest'] = new_io['dest']
        self.update_queue.put({'action': 'PUSH_IO', 'worker_number': worker_number})

    def push_io_to_all(self, new_io):
        for worker_number in range(self.N_workers):
            self.push_io(new_io, worker_number)
    
    # Update model
    def push_model(self, new_model, worker_number):
        self.state['model'] = new_model
        self.update_queue.put({'action': 'PUSH_MODEL', 'worker_number': worker_number})
    
    def push_model_to_all(self, new_model):
        for worker_number in range(self.N_workers):
            self.push_model(new_model, worker_number)
    
    def check_model_state(self, worker_number):
        self.update_queue.put({'action': 'MODEL_STATE', 'worker_number': worker_number})

    def check_all_model_states(self):
        for worker in range(self.N_workers):
            self.check_model_state(worker)
    
    def wait_for_model_load(self):
        self.check_all_model_states()
        while not all(self.model_state.values()):
            sleep(1)
            self.check_all_model_states()

    # Update placement
    def update_placement(self, new_placement):
        self.state['placement_id'] = ''.join([choice(ascii_letters + digits) for _ in range(6)])
        self.state['placement'] = new_placement

    def push_placement(self, worker_number):
        self.update_queue.put({'action': 'PUSH_PLACEMENT', 'worker_number': worker_number})
        self.state['node%s' % worker_number] = {}

    def push_placement_to_all(self):
        for worker_number in range(self.N_workers):
            self.push_placement(worker_number)
    
    # Pull metrics
    def pull_time(self, worker_number):
        self.update_queue.put({'action': 'PULL_STATE', 'worker_number': worker_number})
    
    def pull_all_times(self):
        for worker_number in range(self.N_workers):
            self.pull_time(worker_number)
        while not all(['time' in self.state['node%s'%i] for i in range(self.N_workers)]):
            sleep(1)

    # Bandwidth measurements
    def measure_bandwidth(self, worker_number, remote_host_list):
        self.update_queue.put({'action': 'BANDWIDTH_TEST', 'worker_number': worker_number, 'remote_host_list': remote_host_list})
        self.bandwidth_state['node%s' % worker_number] = False
    
    def measure_all_bandwidths(self):
        for worker in range(self.N_workers - 1):
            remote_host_list = [{'address': self.state['worker_addresses'][n], 'worker_number':n} for n in range(worker + 1, self.N_workers)]
            self.measure_bandwidth(worker, remote_host_list)
    
    def check_bandwidth_state(self, worker_number):
        self.update_queue.put({'action': 'BANDWIDTH_STATE', 'worker_number': worker_number})

    def check_all_bandwidth_states(self):
        for worker in range(self.N_workers - 1):
            self.check_bandwidth_state(worker)
    
    def wait_for_bandwidth_computations(self):
        self.check_all_bandwidth_states()
        while not all(self.bandwidth_state.values()):
            sleep(1)
            self.check_all_bandwidth_states()

    # Claim workers
    def claim_worker(self, worker_number):
        self.update_queue.put({'action': 'CLAIM', 'worker_number': worker_number})
        while not self.bandwidth_state['node%s' % worker_number]:
            sleep(1)
            self.update_queue.put({'action': 'CLAIM', 'worker_number': worker_number})
    
    def claim_workers(self):
        for worker in range(self.N_workers):
            self.claim_worker(worker)

    # New worker
    def init_new_worker(self, worker_number, worker_address):
        self.state['node%s' % worker_number] = {}
        self.bandwidth_state['node%s' % worker_number] = False
        self.model_state['node%s' % worker_number] = False
        self.state['worker_addresses'].append(worker_address) 
        self.N_workers += 1
        
    def init_worker(self, worker_number):
        self.state['node%s' % worker_number] = {}
        self.bandwidth_state['node%s' % worker_number] = False
        self.model_state['node%s' % worker_number] = False
 
    def push_worker_addresses(self, worker):
        self.update_queue.put({'action': 'PUSH_WORKER_ADDRESSES', 'worker_number': worker})

    def push_worker_addresses_all(self):
        for worker in range(self.N_workers):
            self.push_worker_addresses(worker)

    def get_netconf(self, dnnconf, config):
        self.measure_all_bandwidths()
        self.wait_for_bandwidth_computations()
        self.pull_all_times()
        return state_to_netconf(self.state, dnnconf, config)
