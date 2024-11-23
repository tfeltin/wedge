import os
from threading import Thread
from time import time
import onnxruntime as ort

from utils.input_stream import InputStream
from utils.output_stream import OutputStream
from model.preprocess import preprocess
from model.load import create_partitioned_model

providers = os.environ.get('PROVIDERS')
if providers is None:
    providers = ['CPUExecutionProvider']
else:
    providers = eval(providers)


class Partition:
    def __init__(self, layers, input_stream, output_stream, sync_server, config):
        self.start_layer = layers[0]
        self.end_layer = layers[1]

        self.input_name = config['layers'][self.start_layer]['inputs'][0]
        self.output_name = config['layers'][self.end_layer - 1]['outputs'][0]

        self.model_path = create_partitioned_model(config['model'], [self.input_name], [self.output_name])
        self.session = ort.InferenceSession(self.model_path, providers=providers)

        self.input_stream = input_stream
        self.output_stream = output_stream
        self.sync_server = sync_server

        self.partition_thread = Thread(target=self.run, daemon=True)
        self.running = False

        self.is_last_partition = (self.end_layer == len(config['layers']))
        self.input_shape = config['input_shape']

    def run(self):
        self.input_stream.wait_for_connection()

        while self.running:
            # Read image from stream
            image = self.input_stream.read(self.start_layer)
            if image is None:
                break
            # Preproces image if needed
            if self.start_layer == 0:
                image = preprocess(image, self.input_shape)

            # Run partitioned inference
            ti = time()
            preds = self.session.run([self.output_name], {self.input_name: image})[0]
            inference_time = time() - ti
            self.sync_server.update_time(self.start_layer, inference_time)

            # Send result to next partition or output
            self.output_stream.write(self.end_layer, preds)
            
            if not self.is_last_partition:
                self.sync_server.update_transmission_time(self.end_layer, self.output_stream.output_streams[self.end_layer].transmission_time)
    
    def start(self):
        self.running = True
        self.partition_thread.start()

    def stop(self):
        self.running = False
        self.partition_thread.join()


def start_partitions(sync_server, config):
    if config['model'] is None or config['placement'] is None:
        return []

    partition_layers = []
    for i, node in enumerate(config['placement'][1]):
        if node == config['worker_number']:
            partition_layers.append([config['placement'][0][i], config['placement'][0][i+1]])
    input_stream = InputStream(partition_layers, config)
    output_stream = OutputStream(partition_layers, config)

    partitions = [Partition(layers, input_stream, output_stream, sync_server, config) for layers in partition_layers]
    
    for partition in partitions[::-1]:
        partition.start()

    return input_stream, output_stream, partitions


def stop_partitions(input_stream, output_stream, partitions, sync_server):
    if input_stream is not None:
        input_stream.close()
    if output_stream is not None:
        output_stream.close()
    for partition in partitions:
        partition.stop()
    sync_server.reset_telemetry()