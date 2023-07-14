import math
import sys
import time
from multiprocessing import Queue
from threading import Event, Thread
import socket
import pickle
from numpy import asfarray

from vispy import scene
from vispy.color import Color
from vispy.visuals.transforms import STTransform, ChainTransform, MatrixTransform

from src.utils import rotate_yaw_matrix, rotate_pitch_matrix, rotate_roll_matrix, rotation_matrix


class DroneScene:
    def __init__(self, *args, **kwargs):
        self.canvas = scene.SceneCanvas(*args, **kwargs)

        self.view = self.canvas.central_widget.add_view()
        self.view.bgcolor = '#efefef'
        self.view.camera = scene.cameras.FlyCamera()
        self.view.padding = 100
        self.scale = 0.1

        color = Color("#3f51b5")

        self.follow = True

        drone = Drone(self.view.scene)
        self.cube = drone.body

        self.cube.transform = ChainTransform()
        self.cube.transform.append(STTransform())
        self.cube.transform.append(MatrixTransform())
        self.plot = scene.visuals.Line(parent=self.view.scene)

        self.trajectory = []

        scene.visuals.XYZAxis(parent=self.view.scene)

        self.queue = Queue()

        self.fetcher_thread = Thread(target=self.fetcher, daemon=True)
        self.worker_thread = Thread(target=self.worker, daemon=True)

    def move_drone(self, x, y, z):
        self.cube.transform.transforms[0].translate = [x, y, z]
        #if self.follow:
        #    self.view.camera.transform.translate = [x+5, y+5, z+5]

    def rotate_drone(self, yaw, pitch, roll):
        matrix = rotate_pitch_matrix(-yaw).dot(rotate_roll_matrix(pitch).dot(rotate_yaw_matrix(roll).T))
        self.cube.transform.transforms[1].matrix[:3, :3] = matrix

    def fetcher(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(("localhost", 10100))
            print("Connected to server")
            time.sleep(0.2)
            while True:
                print("Send \"GET\"")
                sock.send("GET".encode("utf-8"))
                data = sock.recv(1024)
                if data:
                    pickled_data = pickle.loads(data)
                    print(pickled_data)
                    time.sleep(0.01)
                    self.queue.put(pickled_data)
                else:
                    time.sleep(0.5)

    def worker(self):
        dt = 0.01
        r = 0
        while True:
            if not self.queue.empty():
                data = self.queue.get()

                coords, angles = data
                coords *= self.scale
                coords = asfarray([coords[0], coords[2], coords[1]])

                self.trajectory.append(coords)
                self.plot.set_data(pos=self.trajectory)

                self.move_drone(*coords)
                self.rotate_drone(*angles)

                time.sleep(dt)

    def worker_test(self):
        dt = 0.01
        r = 0
        for i in range(int(10/dt)):
            r = i/100
            x, y, z = r*math.cos(i*math.pi/180), r*math.sin(i*math.pi/180), r
            yaw, pitch, roll = 10 * math.pi/180, 20 * math.pi/180, 30 * math.pi/180
            self.trajectory.append((x, y, z))
            self.plot.set_data(pos=self.trajectory)
            self.move_drone(x, y, z)
            self.rotate_drone(yaw, pitch, roll)
            time.sleep(dt)

    def run(self):
        self.fetcher_thread.start()
        self.worker_thread.start()
        self.canvas.app.run()


class Drone:
    def __init__(self, parent, scale=0.1):
        self.scale = scale
        self.body = scene.visuals.Box(3 * self.scale, 2 * self.scale, 6 * self.scale, color=Color("#3f51b5"),
                                      edge_color="black", parent=parent)

        self.head = scene.visuals.Sphere(radius=1*self.scale, parent=self.body)
        self.head.transform = STTransform(translate=[0, 3*self.scale, 0])

        self.camera = scene.visuals.Box(1 * self.scale, 1 * self.scale, 1 * self.scale, color=Color("#3f51b5"),
                                      edge_color="black", parent=self.body)

        self.camera.transform = STTransform(translate=[0, 0, -0.1])

        self.leg1 = scene.visuals.Box(1 * self.scale, 1 * self.scale, 12 * self.scale, color=Color("red"),
                                      edge_color="black", parent=self.body)
        self.leg2 = scene.visuals.Box(1 * self.scale, 1 * self.scale, 12 * self.scale, color=Color("red"),
                                      edge_color="black", parent=self.body)

        self.leg1.transform = MatrixTransform()
        self.leg2.transform = MatrixTransform()

        self.leg1.transform.rotate(45, (0, 0, 1))
        self.leg2.transform.rotate(-45, (0, 0, 1))

        self.screws1 = scene.visuals.Ellipse((0, 0), radius=(2*self.scale, 2*self.scale), parent=self.leg1)
        self.screws2 = scene.visuals.Ellipse((0, 0), radius=(2*self.scale, 2*self.scale), parent=self.leg1)
        self.screws3 = scene.visuals.Ellipse((0, 0), radius=(2*self.scale, 2*self.scale), parent=self.leg2)
        self.screws4 = scene.visuals.Ellipse((0, 0), radius=(2*self.scale, 2*self.scale), parent=self.leg2)

        self.screws1.transform = STTransform(translate=[6*self.scale, 0, 0.6*self.scale])
        self.screws2.transform = STTransform(translate=[-6*self.scale, 0, 0.6*self.scale])
        self.screws3.transform = STTransform(translate=[6*self.scale, 0, 0.6*self.scale])
        self.screws4.transform = STTransform(translate=[-6*self.scale, 0, 0.6*self.scale])




if __name__ == '__main__' and sys.flags.interactive == 0:
    # canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    #
    # view = canvas.central_widget.add_view()
    # view.bgcolor = '#efefef'
    # view.camera = 'turntable'
    # view.padding = 100
    # scale = 0.1
    #
    # color = Color("#3f51b5")
    #
    # drone = Drone(view.scene)
    #
    # scene.visuals.XYZAxis(parent=view.scene)
    #
    # canvas.app.run()


    drone_scene = DroneScene(keys='interactive', size=(800, 600), show=True)
    drone_scene.run()