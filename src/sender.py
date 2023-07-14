import math
import socket
import pickle
from multiprocessing import Queue
from threading import Thread

import numpy as np


class Sender:
    def __init__(self, addr: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((addr, port))
        print("Waiting client connection...")
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print("Connected by", addr)

        self.queue = Queue()
        self.worker_thread = Thread(target=self.worker, daemon=True)
        self.worker_thread.start()

    def recv(self):
        return self.conn.recv(1024)

    def send(self, data):
        self.conn.send(data)

    def put(self, data):
        self.queue.put(data)

    def get(self):
        if self.queue.empty():
            return None
        return self.queue.get()

    def worker(self):
        while True:
            cmd = self.recv()
            if cmd.decode("utf-8") == "GET":
                print("Receive \"GET\"")
                data = self.queue.get()
                if data is not None:
                    self.send(data)


if __name__ == "__main__":
    sender = Sender("localhost", 10100)

    datas = [np.asfarray([i/100*math.cos(i*np.pi/180), i/100*math.sin(i*np.pi/180), i/100]) for i in range(1000)]

    for data in datas:
        sender.put(pickle.dumps(data))

    while True:
        cmd = sender.recv()
        if cmd.decode("utf-8") == "GET":
            print("Receive \"GET\"")
            data = sender.get()
            if data is not None:
                sender.send(data)
            else:
                break

    #for i in range(5):
    #    data = np.asfarray([i, i+1, i+2])
    #    obj = pickle.dumps(data)
    #    time.sleep(1)
    #    sender.send(obj)
    print("Shutdown server")

