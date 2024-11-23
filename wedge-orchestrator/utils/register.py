import queue
from threading import Thread
import socket


class RegistrationAgent:
    def __init__(self, config, server_agent):
        self.port = int(config['registration_port'])
        self.update_queue = queue.Queue()
        self.registration_request_thread = Thread(target=self.run, daemon=True)
        self.registration_request_thread.start()
        self.server_agent = server_agent

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((b'', self.port))
        sock.listen(1)
 
        while True:
            c, _ = sock.accept()
            _ = c.recv(1024)
            host, _ = c.getpeername()

            # update config with new data and send to sync server
            new_worker = { 
                'worker_address': host
            }
            self.server_agent.add_worker(new_worker)

            c.close()
