import socket
import pickle
import paho.mqtt.client as mqtt
from time import sleep
import numpy as np


class DataStream:
    def __init__(self, host, port, topic):
        self.connected = False
        self.last_message = []
        self.host = host
        self.port = port
        self.mqtt_topic = topic
        
        self.mqtt_client = mqtt.Client("WaaS_test")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.host, port=self.port)
        self.mqtt_client.loop_start()
        while self.connected != True:
            sleep(.1)
        self.mqtt_client.subscribe(self.mqtt_topic)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
        else:
            print("MQTT Client -- Connection failed")
    
    def on_message(self, client, userdata, message):
        inference = eval(message.payload)
        self.last_message = inference['prediction']

    def close(self):
        self.connected = False
        self.mqtt_client.disconnect()
        self.mqtt_client.loop_stop
    
    def read(self):
        return np.asarray(self.last_message)


class InferenceSession:
    
    def __init__(self, model_path, wedge_url):
        self.addr = wedge_url.split(":")[0]
        self.port = int(wedge_url.split(":")[1])
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.addr, self.port))
        sock.sendall(pickle.dumps({'action': 'SYN'}))
        
        data = sock.recv(1024)
        if data != b'ACK':
            print("Error connecting to the Wedge service.")
        sock.close()
        
        self.model = self.load(model_path)
    
    def load(self, model_path):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.addr, self.port))
        sock.sendall(pickle.dumps({'action': 'CONFIG', 'payload': {'model': model_path}}))
        sock.close()

    def run(self, input_url, output_url):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.addr, self.port))
        sock.sendall(pickle.dumps({'action': 'CONFIG', 'payload': {'source': input_url, 'dest': output_url}}))
        sock.close()
        return DataStream(output_url.split(':')[0], int(output_url.split(':')[1]), 'wedge_inference')
    
    def stop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.addr, self.port))
        sock.sendall(pickle.dumps({'action': 'CONFIG', 'payload': {'source': None, 'dest': None}}))
        sock.close()