import queue
from threading import Thread
import socket
import pickle
import queue


class APIServer:
    def __init__(self, config):
        self.port = int(config['api_port'])
        self.source = None
        self.dest = None
        self.model = None
        self.placement = None
        self.update_queue = queue.Queue()
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

            if new_task['action'] == 'CONFIG':
                if 'source' in new_task['payload'] or 'dest' in new_task['payload']:
                    new_source = new_task['payload']['source']
                    new_dest = new_task['payload']['dest']
                    if new_source != self.source or new_dest != self.dest:
                        self.source = new_source
                        self.update_queue.put({'action': 'new_io', 'payload': {'source': new_source, 'dest': new_dest}})
                if 'model' in new_task['payload']:
                    new_model = new_task['payload']['model']
                    if new_model != self.model:
                        self.model = new_model
                        self.update_queue.put({'action': 'new_model', 'payload': new_model})
                if 'placement' in new_task['payload']:
                    new_placement = new_task['payload']['placement']
                    if new_placement != self.placement:
                        self.placement = new_placement
                        self.update_queue.put({'action': 'new_placement', 'payload': new_placement})
            
            elif new_task['action'] == 'COMPUTE_PLACEMENT':
                self.update_queue.put({'action': 'compute_placement'})

            elif new_task['action'] == 'SYN':
                c.sendall(b'ACK')

            c.close()

    def add_worker(self, worker_req):
        self.update_queue.put({'action': 'new_worker','payload': worker_req})

    def wait_for_update(self):
        return self.update_queue.get()
    