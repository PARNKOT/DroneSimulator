import numpy as np
from matplotlib import pyplot as plt

from DroneSimulator.src.utils import get_angles_from_rotation_matrix, rotation_matrix
from DroneSimulator.src.integrator import integrate_linear


if __name__ == "__main__":
    yaw, pitch, roll = 0, 0, 0
    wx, wy, wz = 1*np.pi/180, 0*np.pi/180, 1*np.pi/180
    matrix = rotation_matrix(yaw, pitch, roll).T

    dt = 0.1
    t = 10 / dt

    times = []
    Yaw = []
    Pitch = []
    Roll = []

    for i in range(int(t)):
        times.append(i*dt)
        yaw, pitch, roll = get_angles_from_rotation_matrix(matrix)
        Yaw.append(yaw * 180 / np.pi)
        Pitch.append(pitch * 180 / np.pi)
        Roll.append(roll * 180 / np.pi)

        speeds_matrix = np.asfarray([[0, -wz, wy],
                                     [wz, 0, -wx],
                                     [-wy, wx, 0]])
        matrix = integrate_linear(matrix, matrix.dot(speeds_matrix), dt)

    plt.plot(times, Yaw)
    plt.plot(times, Pitch)
    plt.plot(times, Roll)
    plt.legend(["Yaw", "Pitch", "Roll"])
    plt.grid()
    plt.show()