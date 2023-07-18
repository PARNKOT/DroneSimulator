import numpy as np
import pickle
import math
import random

from drone_model import DroneMini
from controls import Controller, MeasuredParams
from sender import Sender
from utils import Rad_TO_DEGREES


drone_params = {
    "mass": 1,
    "b_engine": 26.5e-6,
    "d_engine": 0.6e-6,
    "shoulders": [0.15],
    "tensor": np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.1, 0.0],
                         [0.0, 0.0, 0.1]], dtype="float64"),
}


GYRO_ERROR = 1 # degrees per second
LINEAR_SPEED_ERROR = 0.05


def trajectory(t):
    if t > 30:
        return (t, 200, 500)
    return (t, 200, 0.5 * t ** 2)

def main():
    sender = Sender("localhost", 10100)

    drone = DroneMini(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(0 * np.pi / 180, 0 * np.pi / 180, 0)

    controller = Controller()

    controller.angular_speeds_limits = np.asfarray([200, 200, 200]) / Rad_TO_DEGREES
    controller.angles_limits = np.asfarray([math.inf, 30, 30]) / Rad_TO_DEGREES
    controller.linear_speeds_limits = [15, 6, 15]

    dt = 0.01  # second
    t = int(40 / dt)

    X_des = 200
    Y_des = 100
    Z_des = 300
    Yaw_des = 0 * np.pi / 180

    for i in range(t):
        measurements = MeasuredParams()

        measurements.Rotation_speeds = drone.rotation_speeds
        measurements.Linear_speeds = drone.linear_speeds
        measurements.Angles = drone.angles
        measurements.Coords = drone.linear_coords

        error1 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error2 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error3 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        measurements.Rotation_speeds[0] += error1
        measurements.Rotation_speeds[1] += error2
        measurements.Rotation_speeds[2] += error3

        error4 = random.gauss(0, LINEAR_SPEED_ERROR)
        error5 = random.gauss(0, LINEAR_SPEED_ERROR)
        error6 = random.gauss(0, LINEAR_SPEED_ERROR)
        measurements.Linear_speeds[0] += error4
        measurements.Linear_speeds[1] += error5
        measurements.Linear_speeds[2] += error6

        state = np.asfarray([measurements.Coords, measurements.Angles])

        sender.put(pickle.dumps(state))

        #X_des, Y_des, Z_des = trajectory(i * dt)

        engines_speeds = controller.handle(Yaw_des, [X_des, Y_des, Z_des], measurements, dt)

        drone.integrate(engines_speeds, dt)


if __name__ == "__main__":
    main()
