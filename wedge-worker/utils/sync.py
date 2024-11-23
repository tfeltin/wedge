import queue
from threading import Thread
import socket
import pickle
import time
import queue

from utils.metrics import BandwidthAgent


class SyncAgent:
    def __init__(self, config):

        self.state = {'placement_id': config['placement_id'],
                      'placement': config['placement'],
                      'telemetry_placement': config['placement'],
                      'worker_number': config['worker_number'],
                      'worker_addresses': config['worker_addresses'],
                      'model': config['model'],
                      'source': config['source'],
                      'dest': config['dest'],
                      'time': {},
                      'transmission_time': {},
                      'last_update':{},
                     }
        
        self.port = int(config['sync_port'])
        self.update_queue = queue.Queue()
        self.bandwidth_agent = BandwidthAgent(config)
        self.model_loaded = False
        self.tcp_server_thread = Thread(target=self.run, daemon=True)
        self.tcp_server_thread.start()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((b'', self.port))
        sock.listen(1)

        while True:
            c, _ = sock.accept()
            data = c.recv(1024)
            new_task = pickle.loads(data)

            if new_task['action'] == 'PULL_STATE':
                self.state['bandwidths'] = self.bandwidth_agent.bandwidths
                serialized_data = pickle.dumps(self.state)
                c.send(serialized_data)

            elif new_task['action'] == 'CLAIM':
                message = new_task['payload']
                change_flag = False
                if 'worker_number' in message.keys() and message['worker_number'] != self.state['worker_number']:
                    print("Sync Server -- Worker claimed as node #%s" % message['worker_number'])
                    change_flag = True
                    self.state['worker_number'] = message['worker_number']
                    self.bandwidth_agent.worker_number = message['worker_number']
                if 'worker_addresses' in message.keys() and message['worker_addresses'] != self.state['worker_addresses']:
                    change_flag = True
                    self.state['worker_addresses'] = message['worker_addresses']

                serialized_data = pickle.dumps({'worker_number' : self.state['worker_number']})
                c.send(serialized_data)
                if change_flag:
                    self.update_queue.put(new_task)
            
            elif new_task['action'] == 'PUSH_IO':
                message = new_task['payload']
                print("Sync Server -- New I/O -- Source: %s -- Dest: %s" % (message['source'], message['dest']))
                self.state['source'] = message['source']
                self.state['dest'] = message['dest']
                self.update_queue.put(new_task)
            
            elif new_task['action'] == 'PUSH_PLACEMENT':
                message = new_task['payload']
                if 'placement_id' in message.keys() and 'placement' in message.keys() and self.state['placement_id'] != message['placement_id']:
                    print("Sync Server -- New placement %s : %s" % (message['placement_id'], message['placement']))
                    self.state['placement_id'] = message['placement_id']
                    self.state['placement'] = message['placement']
                    self.update_queue.put(new_task)

            elif new_task['action'] == 'PUSH_MODEL':
                new_model = new_task['payload']
                if self.state['model'] != new_model:
                    print("Sync Server -- Received model '%s' " % new_model)
                    self.state['model'] = new_model
                    self.update_queue.put(new_task)
            
            elif new_task['action'] == "BANDWIDTH_TEST":
                remote_host_list = new_task['payload']
                for remote_host in remote_host_list:
                    if remote_host['address'] not in self.state['transmission_time'].keys():
                        self.bandwidth_agent.get_bw(remote_host['address'], remote_host['worker_number'])
            
            elif new_task['action'] == "BANDWIDTH_STATE":
                serialized_data = pickle.dumps({'worker_number': self.state['worker_number'], 'state' : self.bandwidth_agent.done})
                c.send(serialized_data)
            
            elif new_task['action'] == "MODEL_STATE":
                serialized_data = pickle.dumps({'worker_number': self.state['worker_number'], 'state' : self.model_loaded})
                c.send(serialized_data)

            elif new_task['action'] == 'WORKER_ADDRESSES':
                self.state['worker_addresses'] = new_task['payload']
                self.update_queue.put(new_task)

            c.close()


    def update_time(self, start_layer, new_time, smoothness=.1):
        t = time.time()
        # Update compute time
        if start_layer in self.state['time'].keys():
            self.state['time'][start_layer] = (1 - smoothness) * self.state['time'][start_layer] + smoothness * new_time
        else:
            self.state['time'][start_layer] = new_time
        # Update last update time
        self.state['last_update'][start_layer] = t
   
    def update_transmission_time(self, end_layer, new_time, smoothness=.2):
        if end_layer in self.state['transmission_time'].keys():
            self.state['transmission_time'][end_layer] = (1 - smoothness) * self.state['transmission_time'][end_layer] + smoothness * new_time
        else:
            self.state['transmission_time'][end_layer] = new_time

    def wait_for_update(self):
        return self.update_queue.get()
    
    def reset_telemetry(self):
        if self.state['placement'] is not None and self.state['worker_number'] in self.state['placement'][1]:
            self.state['time'] = {}
            self.state['transmission_time'] = {}
            self.state['last_update']= {}
            self.state['telemetry_placement'] = self.state['placement']
