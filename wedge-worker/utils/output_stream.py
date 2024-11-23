import queue
from threading import Thread
import socket
import pickle
import struct
from random import choice
from string import ascii_letters, digits
from time import sleep, time
from json import dumps
import paho.mqtt.client as mqtt
import re


class TCPClient:
    def __init__(self, next_partition, layer, config):
        self.placement_id = config['placement_id']
        self.address = config['worker_addresses'][next_partition]
        self.port = int(config['tcp_port'])
        self.layer = layer
        self.message_queue = queue.Queue()
        self.connected = 0
        self.transmission_time = 0
        self.tcp_client_thread = Thread(target=self.run, daemon=True)
        self.tcp_client_thread.start()
    
    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while self.connected == 0:
            try:
                self.sock.connect((self.address, self.port))
                self.sock.sendall(self.placement_id.encode('utf-8'))
                ack = self.sock.recv(1024)
                if ack == b'OK.':
                    self.connected = 1
                    print("TCP Client %s -- Connected to %s:%s..." % (self.placement_id, self.address, self.port))
                else:
                    continue
            except (ConnectionRefusedError, ConnectionResetError):
                print("TCP Client %s -- Connection refused, trying again." % self.placement_id)
                sleep(1)
            except OSError:
                print("TCP Client %s -- Socket already in use, trying again." % self.placement_id)
                sleep(1)
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        while self.connected == 1:
            data = self.message_queue.get()
            if data is None:
                return
            serialized_data = pickle.dumps({'layer': self.layer, 'data' : data}, protocol=2)
            packet_size = struct.pack("L", len(serialized_data))
            try:
                t0 = time()
                self.sock.sendall(packet_size + serialized_data)
                tf = time() - t0
                if self.transmission_time == 0:
                    self.transmission_time = tf
                else:
                    self.transmission_time = self.transmission_time * .5 + tf * .5
            except:
                self.connected = 0
                self.sock.close()
                print("TCP Client %s -- Connection closed by server." % self.placement_id)
                return

    def write(self, data):
        if not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                pass
        self.message_queue.put(data)
    
    def wait_for_connection(self):
        while self.connected == 0:
            sleep(1)
        if self.connected < 0:
            raise ConnectionRefusedError("TCP Client %s -- Connection to %s:%s refused." % (self.placement_id, self.address, self.port))

    def close(self):
        if self.connected == 1:
            self.connected = 0
            self.sock.close()
        self.write(None)
        self.tcp_client_thread.join()
        print("TCP Client %s -- Connection closed." % self.placement_id)
    

class TCPAppClient:
    def __init__(self, config):
        port = re.findall(r':(\d+)', config['dest'])
        if len(port) > 0:
            self.host = re.findall(r'(.*):\d+', config['dest'])[0]
            self.port = int(port[0])
        else:
            self.host, self.port = config['dest'], 80
        self.placement_id = config["placement_id"]
        self.message_queue = queue.Queue()
        self.connected = 0
        self.tcp_client_thread = Thread(target=self.run, daemon=True)
        self.tcp_client_thread.start()
        self.framerate = 0

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while self.connected == 0:
            try:
                self.sock.connect((self.host, self.port))
                self.connected = 1
                print("TCP App Client  --  Connected to %s:%s", self.host, self.port)
            except (ConnectionRefusedError, ConnectionResetError):
                print("TCP App Client  -- Connection refused, trying again.")
                sleep(1)
        
        t0 = time()
        while self.connected == 1:
            data = self.message_queue.get()
            t1 = time()
            if self.framerate > 0:
                self.framerate = .5 * self.framerate + .5 / (t1 - t0)
            else:
                self.framerate = 1 / (t1 - t0)
            if data is None:
                sleep(1)
                continue
            serialized_data = pickle.dumps({'placement_id' : self.placement_id, 'framerate': round(self.framerate, 1), 'prediction' : data.tolist()})
            try:
                self.sock.sendall(serialized_data)
            except Exception as e:
                self.connected = 0
                self.sock.close()
                print("TCP App Client-- Connection closed by server: %s" % e)

    def write(self, data):
        if not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                pass
        self.message_queue.put(data)
    
    def wait_for_connection(self):
        while self.connected == 0:
            sleep(1)
        if self.connected < 0:
            raise ConnectionRefusedError("TCP App Client %s -- Connection to %s:%s refused." % (self.placement_id, self.host, self.port))

    def close(self):
        self.connected = 0
        self.sock.close()
        self.write(None)
        self.tcp_client_thread.join()
        print("TCP App Client %s -- Connection closed." % self.placement_id)


class MQTTClient:
    def __init__(self, config):
        self.placement_id = config["placement_id"]
        self.connected = True
        self.mqtt_queue = queue.Queue() 
        self.mqtt_topic = config['mqtt_topic']
        self.framerate = 0

        mqtt_dest = config['dest'].replace('mqtt://','')
        if ':' in mqtt_dest:
            host, port = mqtt_dest.split(':')
            port = int(port)
        else:
            host, port = mqtt_dest, 1883
        
        self.mqtt_client_id = "".join([choice(ascii_letters + digits) for _ in range(10)])
        self.mqtt_client = mqtt.Client("wedge_%s" % self.mqtt_client_id)
        try:
            self.mqtt_client.connect(host, port=port)
            print("MQTT Client %s -- Connected to broker" % self.placement_id)
        except ConnectionRefusedError:
            raise ConnectionRefusedError("MQTT Client %s -- Connection to broker refused." % self.placement_id)
        
        self.mqtt_thread = Thread(target=self.update, daemon=True)
        self.mqtt_thread.start()

    def update(self):
        t0 = time()
        while self.connected:
            mqtt_new_message = self.mqtt_queue.get()
            t1 = time()
            if self.framerate > 0:
                self.framerate = .5 * self.framerate + .5 / (t1 - t0)
            else:
                self.framerate = 1 / (t1 - t0)
            try:
                self.mqtt_client.publish(self.mqtt_topic, dumps({'placement_id' : self.placement_id, 'framerate': round(self.framerate, 1), 'prediction' : mqtt_new_message.tolist()}))
                t0 = t1
            except TypeError as e:
                print("MQTT Client -- Wrongly formatted MQTT message, dropping.")
                print(e)
    
    def write(self, mqtt_new_message):
        self.mqtt_queue.put(mqtt_new_message)

    def close(self):
        self.connected = False
        print("MQTT Client %s -- Connection closed." % self.placement_id)


class OutputStream:
    def __init__(self, partitions, config):
        self.output_streams = {}
        if len(partitions) > 0:
            for i, end_layer in enumerate([p[1] for p in partitions]):
                if end_layer < config['placement'][0][-1]:
                    next_partition = config['placement'][1][config['placement'][0].index(partitions[i][1])]
                    self.output_streams[end_layer] = TCPClient(next_partition, end_layer, config)
                elif "mqtt://" in config['dest']:
                    self.output_streams[end_layer] = MQTTClient(config)
                else:
                    self.output_streams[end_layer] = TCPAppClient(config)

    def write(self, end_layer, message):
        return self.output_streams[end_layer].write(message)

    def close(self):
        for output_stream in list(self.output_streams.values())[::-1]:
            output_stream.close()
