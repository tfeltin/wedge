from threading import Thread
import queue
import cv2
from time import sleep


class RTSPIn:
    def __init__(self, config):
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
        print('RTSP Client -- Successfully opened %s (%gx%g at %.2f FPS).' % (self.source, self.w, self.h, self.fps))

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

    def read(self):
        if self.connected:
            return self.in_queue.get()
    
    def wait_for_connection(self):
        while not self.connected:
            sleep(1)

    def close(self):
        self.connected = False
        self.input_thread.join()
        self.in_queue.put(None)
        print("RTSP Client -- Connection closed.")

