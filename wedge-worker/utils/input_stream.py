from threading import Thread
import queue
import socket
import struct
import pickle
import cv2
from time import sleep


class RTSPIn:
    def __init__(self, config):
        self.placement_id = config['placement_id']
        self.in_queue = queue.Queue()
        self.source = config['source']
        self.connected = False

        self.input_thread = Thread(target=self.update, daemon=True)
        self.input_thread.start()
    
    def update(self):
        self.cap = cv2.VideoCapture(self.source)
        assert self.cap.isOpened(), 'RTSP Client -- Failed to open %s' % self.source
        self.connected = True
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) % 100
        print('RTSP Client %s -- Successfully opened %s (%gx%g at %.2f FPS).' % (self.placement_id, self.source, self.w, self.h, self.fps))

        while self.connected:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.in_queue.empty():
                try:
                    self.in_queue.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.in_queue.put(frame)
        self.cap.release()

    def read(self, layer):
        if self.connected:
            return self.in_queue.get()
    
    def wait_for_connection(self):
        while not self.connected:
            sleep(1)

    def close(self):
        self.connected = False
        self.input_thread.join()
        self.in_queue.put(None)
        print("RTSP Client %s -- Connection closed." % self.placement_id)


class TCPServer:
    def __init__(self, layers, config):
        self.placement_id = config['placement_id']
        self.port = int(config['tcp_port'])
        self.received_msg_queues = {}
        self.tcp_server_threads = {}
        self.connected = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((b'', self.port))
        self.sock.listen(1)
        print("TCP Server %s -- Waiting for connection..." % self.placement_id)

        for l in layers:
            self.received_msg_queues[l] = queue.LifoQueue()
            self.connected[l] = False
            self.tcp_server_threads[l] = Thread(target=self.run, args=(l,), daemon=True)
            self.tcp_server_threads[l].start()
        
    def run(self, l):
        remote_placement_id = ''
        while remote_placement_id != self.placement_id.encode('utf-8'):
            c, a = self.sock.accept()
            remote_placement_id = c.recv(1024)
            if remote_placement_id != self.placement_id.encode('utf-8'):
                print("TCP Server %s -- Wrong ID from %s:%s -- remote_placement_id=%s, self.placement_id=%s" % (self.placement_id, *a, remote_placement_id, self.placement_id.encode('utf-8')))
                c.sendall(b'Wrong ID.')
                c.close()
            else:
                print("TCP Server %s -- Connected to %s:%s." % (self.placement_id, *a))
                c.sendall(b'OK.')
                break
        
        self.connected[l] = True
        data = b''

        factor = 8

        while self.connected[l]:
            payload_size = struct.calcsize("L")
            while len(data) < payload_size:
                data += c.recv(max(1024, payload_size//factor))
                if data == b'':
                    self.connected[l] = False
                    c.close()
                    print("TCP Server %s -- Connection closed by client." % self.placement_id)
                    return
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]

            msg_size = struct.unpack("L", packed_msg_size)[0]

            # Retrieve all data based on message size
            while len(data) < msg_size:
                packet = c.recv(max(1024, msg_size//factor))
                if packet == b'':
                    self.connected[l] = False
                    c.close()
                    print("TCP Server  %s-- Connection closed by client." % self.placement_id)
                    return
                data += packet

            frame_data = data[:msg_size]
            data = data[msg_size:]

            unserialized_input = pickle.loads(frame_data)
            layer = int(unserialized_input['layer'])
            unserialized_data = unserialized_input['data']

            if layer in list(self.received_msg_queues.keys()):
                if not self.received_msg_queues[layer].empty():
                    try:
                        self.received_msg_queues[layer].get_nowait()   # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                self.received_msg_queues[layer].put(unserialized_data)
            else:
                print("Received wrong layer (%s not in %s)" % (layer, list(self.received_msg_queues.keys())))
        
    def read(self, layer):
        if self.connected:
            return self.received_msg_queues[layer].get()
        else:
            return None
    
    def wait_for_connection(self):
        while not all(self.connected.values()):
            sleep(1)
    
    def close(self):
        for layer in self.connected:
            self.connected[layer] = False
        for t in self.tcp_server_threads.values():
            t.join()
        self.sock.close()
        for layer in self.received_msg_queues:
            self.received_msg_queues[layer].put(None)
        print("TCP Server %s -- Connection closed." % self.placement_id)


class InputStream:
    def __init__(self, partitions, config):
        self.input_streams = {}
        if len(partitions) > 0:
            if partitions[0][0] == 0:
                self.input_streams[0] = RTSPIn(config)

            tcp_start_layers = [p[0] for p in partitions if p[0] > 0]
            if len(tcp_start_layers) > 0:
                tcp_server = TCPServer(tcp_start_layers, config)
                for start_layer in tcp_start_layers:
                    self.input_streams[start_layer] = tcp_server
        
    def read(self, layer):
        return self.input_streams[layer].read(layer)
    
    def wait_for_connection(self):
        for input_stream in self.input_streams.values():
            input_stream.wait_for_connection()
    
    def close(self):
        for input_stream in list(self.input_streams.values())[::-1]:
            input_stream.close()
