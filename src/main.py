import numpy as np
import pickle

from drone_model import DroneModel
from controls import Controller, MeasuredParams
from sender import Sender


drone_params = {
    "mass": 0.8,
    "b_engine": 26.5e-6,
    "d_engine": 0.6e-6,
    "shoulder": 0.15,
    "tensor": np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.1, 0.0],
                         [0.0, 0.0, 0.1]], dtype="float64"),
}


def trajectory(t):
    if t > 30:
        return (t, 200, 500)
    return (t, 200, 0.5 * t ** 2)

def main():
    sender = Sender("localhost", 10100)

    drone = DroneModel(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(0 * np.pi / 180, 0 * np.pi / 180, 0)

    controller = Controller()

    dt = 0.01  # second
    t = int(30 / dt)

    X_des = 300
    Y_des = 50
    Z_des = 300
    Yaw_des = 90 * np.pi / 180

    for i in range(t):
        measurements = MeasuredParams()

        measurements.Rotation_speeds = drone.rotation_speeds
        measurements.Linear_speeds = drone.linear_speeds
        measurements.Angles = drone.angles
        measurements.Coords = drone.linear_coords

        state = np.asfarray([measurements.Coords, measurements.Angles])

        sender.put(pickle.dumps(state))

        #X_des, Y_des, Z_des = trajectory(i * dt)

        engines_speeds = controller.handle(Yaw_des, [X_des, Y_des, Z_des], measurements, drone.rotation_matrix, dt)

        drone.integrate(engines_speeds, dt)


if __name__ == "__main__":
    main()
