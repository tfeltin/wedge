import os
import numpy as np
import queue
import socket
from threading import Thread
from time import time
import onnx
import onnxruntime as ort

from model.profiler import onnx_shape_to_array, create_random_data

providers = os.environ.get('PROVIDERS')
if providers is None:
    providers = ['CPUExecutionProvider']
else:
    providers = eval(providers)


def benchmark_node(model_path, N_iter=50):
    print("Wedge Worker -- Benchmarking node")

    model = onnx.load(model_path)
    sess = ort.InferenceSession(model_path, providers=providers)

    shape_arr = onnx_shape_to_array(model.graph.input[0].type.tensor_type.shape)
    output_names = [o.name for o in model.graph.output]
    
    # Runs the inference
    inference_time = 0
    for _ in range(N_iter):
        input_dict = {model.graph.input[0].name : create_random_data(shape_arr, np.float32, 0, 1, None)}
        t0 = time()
        sess.run(output_names, input_dict)
        if inference_time == 0:
            inference_time = time() - t0
        else:
            inference_time = inference_time*.9 + (time() - t0)*.1
    return inference_time


class BandwidthAgent:
    def __init__(self, config):
        self.worker_number = config['worker_number']
        self.task_queue = queue.Queue()
        self.bandwidths = {}
        self.server_thread = Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        self.client_thread = Thread(target=self.run_client, daemon=True)
        self.client_thread.start()
        self.done = True

    def run_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 5201))
        s.listen(1)

        while True:
            conn, _ = s.accept()
            print('Bandwidth Test -- Incoming bandwidth test')
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                del data
            conn.send(b"OK.\n")
            conn.close()

    def run_client(self):
        while True:
            task = self.task_queue.get()
            print("Bandwidth Test -- Starting client bandwidth test with %s" % task['remote_host'])
            buffer_size = 1024 # 1kB
            count = 10000 # 10 MB
            testdata = 'x' * (buffer_size-1) + '\n'
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((task['remote_host'], 5201))
            t3 = time()
            i = 0
            while i < count:
                i = i+1
                s.send(bytearray(testdata,"utf-8"))
            s.shutdown(1)
            _ = s.recv(buffer_size)
            t5 = time()
            bps = (buffer_size * count) / (t5 - t3)
            Mbps = bps / 1000000
            print ('Bandwidth Test -- Bandwidth : %.2fMbps in %.1fs' % (Mbps, (t5 - t3)))
            
            self.bandwidths[(self.worker_number, int(task['worker_number']))] = bps

            # Notify orchestrator when done
            if self.task_queue.empty():
                self.done = True

    def get_bw(self, remote_host, worker_number):
        self.task_queue.put({'remote_host': remote_host, 'worker_number': worker_number})
        self.done = False
