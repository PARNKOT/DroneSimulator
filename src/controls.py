import typing

import numpy as np

from DroneSimulator.src.pid import PID
from DroneSimulator.src.integrator import IntegratorFactory, LINEAR

drone_params = {
    "mass": 0.15,
    "b_engine": 7e-7,
    "d_engine": 7e-8,
    "shoulder": 0.15,
    "tensor": np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.2, 0.0],
                         [0.0, 0.0, 0.1]], dtype="float64"),
}

control1_params = {
    "k_p": 800,
    "k_i": 0.5,
    "k_d": 2,
}

control2_params = {
    "k_p_angles": 5.5,
    "k_i_angles": 1.8,
    "k_d_angles": 2,
}

INTEGRATOR_TYPE = LINEAR


class RotationSpeedsControlLoop:
    k_p, k_i, k_d = 250, 0.5, 2

    def __init__(self, k_p, k_i, k_d):
        self.__pid_wx = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_wy = PID(k_p+150, k_i+20, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_wz = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))

    def set_initial_state(self, init_speeds: typing.Sequence):
        self.__pid_wx.reset(init_speeds[0])
        self.__pid_wy.reset(init_speeds[1])
        self.__pid_wz.reset(init_speeds[2])

    def handle(self, thrust_cmd, speeds_des: typing.Sequence, speeds_cur: typing.Sequence, dt):
        wx_des, wy_des, wz_des = speeds_des
        wx_cur, wy_cur, wz_cur = speeds_cur

        wx_cmd = self.__pid_wx.calculate(wx_des-wx_cur, dt)
        wy_cmd = self.__pid_wy.calculate(wy_des-wy_cur, dt)
        wz_cmd = self.__pid_wz.calculate(wz_des-wz_cur, dt)

        return np.asfarray([thrust_cmd, wx_cmd, wy_cmd, wz_cmd])


class AnglesControlLoop:
    k_p_angles, k_i_angles, k_d_angles = 2.3, 0.1, 2
    k_p_yaw, k_i_yaw, k_d_yaw = -10.05, -0.5, -0.1

    def __init__(self, k_p_angles, k_i_angles, k_d_angles):
        self.__pid_yaw = PID(-1.1, -0.35, -0.1, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_pitch = PID(k_p_angles, k_i_angles, k_d_angles, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_roll = PID(k_p_angles, k_i_angles, k_d_angles, IntegratorFactory.create(INTEGRATOR_TYPE))

    def set_initial_state(self, init_angles: typing.Sequence):
        self.__pid_yaw.reset(init_angles[0])
        self.__pid_pitch.reset(init_angles[1])
        self.__pid_roll.reset(init_angles[2])

    def handle(self, thrust_cmd, angles_des: typing.Sequence, angles_cur: typing.Sequence, dt):
        yaw_des, pitch_des, roll_des = angles_des
        yaw_cur, pitch_cur, roll_cur = angles_cur

        wx_cmd = self.__pid_roll.calculate(roll_des - roll_cur, dt)
        wy_cmd = self.__pid_yaw.calculate(yaw_des-yaw_cur, dt)
        wz_cmd = self.__pid_pitch.calculate(pitch_des-pitch_cur, dt)

        return np.asfarray([thrust_cmd, wx_cmd, wy_cmd, wz_cmd])


class CoordsControlLoop:
    def __init__(self, k_p, k_i, k_d):
        self.__pid_thrust = PID(7, 0.4, 20, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_x = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_z = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))

    def set_initial_state(self, init_state: typing.Sequence):
        self.__pid_thrust.reset(init_state[0])
        self.__pid_x.reset(init_state[1])
        self.__pid_z.reset(init_state[2])

    def handle(self, coords_des: typing.Sequence, coords_cur: typing.Sequence, dt):
        x_des, y_des, z_des = coords_des
        x_cur, y_cur, z_cur = coords_cur

        thrust_cmd = self.__pid_thrust.calculate(y_des - y_cur, dt)
        x_cmd = self.__pid_x.calculate(x_des-x_cur, dt)
        z_cmd = self.__pid_z.calculate(z_des-z_cur, dt)

        return np.asfarray([x_cmd, thrust_cmd, z_cmd])


def model_rotation_speeds():
    from matplotlib import pyplot as plt
    from DroneSimulator.src.drone_model import DroneModel
    from DroneSimulator.src.command_mixer import mix_commands_cross
    from DroneSimulator.src.utils import saturate
    drone = DroneModel(**drone_params)

    k_p, k_i, k_d = 800, 0.5, 2
    #k_p_angles, k_i_angles, k_d_angles = 2.3, 0.1, 2

    control1 = RotationSpeedsControlLoop(k_p, k_i, k_d)
    #control2 = AnglesControlLoop(k_p_angles, k_i_angles, k_d_angles)

    dt = 0.1  # second

    Pdes = 290
    Wdes_x = 10 * np.pi / 180
    Wdes_y = 10 * np.pi / 180
    Wdes_z = 10 * np.pi / 180

    times = []

    Wx = []
    Wy = []
    Wz = []

    Yaw = []
    Pitch = []
    Roll = []

    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    for i in range(100):
        times.append(i * dt)

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speeds
        Yaw_cur, Pitch_cur, Roll_cur = drone.angles

        Yaw.append(Yaw_cur * 180 / np.pi)
        Pitch.append(Pitch_cur * 180 / np.pi)
        Roll.append(Roll_cur * 180 / np.pi)

        x, y, z = drone.linear_coords
        X.append(x)
        Y.append(y)
        Z.append(z)

        Wx.append(Wcur_x * 180 / np.pi)
        Wy.append(Wcur_y * 180 / np.pi)
        Wz.append(Wcur_z * 180 / np.pi)

        speeds_des = [Wdes_x, Wdes_y, Wdes_z]

        engines_speeds = mix_commands_cross(control1.handle(Pdes, speeds_des, [Wcur_x, Wcur_y, Wcur_z], dt))

        for i, speed in enumerate(engines_speeds):
            engines_speeds[i] = saturate(speed, 2500)
            speeds[i].append(saturate(speed, 2500))

        drone.integrate(engines_speeds, dt)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    ax1.plot(times, Wx)
    ax1.plot(times, Wy)
    ax1.plot(times, Wz)
    ax1.grid()
    ax1.legend(["Wx", "Wy", "Wz"])

    ax2.plot(times, speeds[0])
    ax2.plot(times, speeds[1])
    ax2.plot(times, speeds[2])
    ax2.plot(times, speeds[3])
    ax2.legend(["W1", "W2", "W3", "W4"])
    ax2.grid()

    ax3.plot(times, Yaw)
    ax3.plot(times, Pitch)
    ax3.plot(times, Roll)
    ax3.legend(["Yaw", "Pitch", "Roll"])
    ax3.grid()

    ax4.plot(times, X)
    ax4.plot(times, Y)
    ax4.plot(times, Z)
    ax4.legend(["X", "Y", "Z"])
    ax4.grid()

    plt.show()


def model_rotation():
    from matplotlib import pyplot as plt
    from DroneSimulator.src.drone_model import DroneModel
    from DroneSimulator.src.command_mixer import mix_commands_cross
    from DroneSimulator.src.utils import saturate
    drone = DroneModel(**drone_params)

    k_p_angles, k_i_angles, k_d_angles = 5.5, 1.8, 2

    control1 = RotationSpeedsControlLoop(**control1_params)
    control2 = AnglesControlLoop(k_p_angles, k_i_angles, k_d_angles)

    dt = 0.1  # second

    Pdes = 290
    Yaw_des = 10 * np.pi / 180
    Pitch_des = 10 * np.pi / 180
    Roll_des = 10 * np.pi / 180

    times = []

    Wx = []
    Wy = []
    Wz = []

    Yaw = []
    Pitch = []
    Roll = []

    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    for i in range(100):
        times.append(i * dt)

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speeds
        Yaw_cur, Pitch_cur, Roll_cur = drone.angles

        Yaw.append(Yaw_cur * 180 / np.pi)
        Pitch.append(Pitch_cur * 180 / np.pi)
        Roll.append(Roll_cur * 180 / np.pi)

        x, y, z = drone.linear_coords
        X.append(x)
        Y.append(y)
        Z.append(z)

        Wx.append(Wcur_x * 180 / np.pi)
        Wy.append(Wcur_y * 180 / np.pi)
        Wz.append(Wcur_z * 180 / np.pi)

        thrust, *speeds_des = control2.handle(Pdes, [Yaw_des, Pitch_des, Roll_des],
                                     [Yaw_cur, Pitch_cur, Roll_cur], dt)

        engines_speeds = mix_commands_cross(control1.handle(Pdes, speeds_des,
                                                            [Wcur_x, Wcur_y, Wcur_z], dt))

        for i, speed in enumerate(engines_speeds):
            engines_speeds[i] = saturate(speed, 2500)
            speeds[i].append(saturate(speed, 2500))

        drone.integrate(engines_speeds, dt)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    ax1.plot(times, Wx)
    ax1.plot(times, Wy)
    ax1.plot(times, Wz)
    ax1.grid()
    ax1.legend(["Wx", "Wy", "Wz"])

    ax2.plot(times, speeds[0])
    ax2.plot(times, speeds[1])
    ax2.plot(times, speeds[2])
    ax2.plot(times, speeds[3])
    ax2.legend(["W1", "W2", "W3", "W4"])
    ax2.grid()

    ax3.plot(times, Yaw)
    ax3.plot(times, Pitch)
    ax3.plot(times, Roll)
    ax3.legend(["Yaw", "Pitch", "Roll"])
    ax3.grid()

    ax4.plot(times, X)
    ax4.plot(times, Y)
    ax4.plot(times, Z)
    ax4.legend(["X", "Y", "Z"])
    ax4.grid()

    plt.show()


def model_coords():
    from matplotlib import pyplot as plt
    from DroneSimulator.src.drone_model import DroneModel
    from DroneSimulator.src.command_mixer import mix_commands_cross
    from DroneSimulator.src.utils import saturate, saturate_min_max
    drone = DroneModel(**drone_params)
    drone.linear_coords = np.asfarray([0, 0, 0])

    k_p, k_i, k_d = 10, 0, 0

    control1 = RotationSpeedsControlLoop(**control1_params)
    control2 = AnglesControlLoop(**control2_params)
    control3 = CoordsControlLoop(k_p, k_i, k_d)

    dt = 0.1  # second

    thrust_const = 0
    X_des = 0
    Y_des = 50
    Z_des = 0
    Yaw_des = 0

    times = []

    Wx = []
    Wy = []
    Wz = []

    Yaw = []
    Pitch = []
    Roll = []

    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    for i in range(1000):
        times.append(i * dt)

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speeds
        X_cur, Y_cur, Z_cur = drone.linear_coords
        Yaw_cur, Pitch_cur, Roll_cur = drone.angles

        Yaw.append(Yaw_cur * 180 / np.pi)
        Pitch.append(Pitch_cur * 180 / np.pi)
        Roll.append(Roll_cur * 180 / np.pi)

        x, y, z = drone.linear_coords
        X.append(x)
        Y.append(y)
        Z.append(z)

        Wx.append(Wcur_x * 180 / np.pi)
        Wy.append(Wcur_y * 180 / np.pi)
        Wz.append(Wcur_z * 180 / np.pi)

        x_cmd, thrust, z_cmd = control3.handle([X_des, Y_des, Z_des], [X_cur, Y_cur, Z_cur], dt)

        thrust, *speeds_des = control2.handle(thrust, [Yaw_des, x_cmd, z_cmd],
                                     [Yaw_cur, Pitch_cur, Roll_cur], dt)

        thrust += thrust_const

        engines_speeds = mix_commands_cross(control1.handle(thrust, speeds_des,
                                                            [Wcur_x, Wcur_y, Wcur_z], dt))

        for i, speed in enumerate(engines_speeds):
            speeds[i].append(saturate_min_max(speed, 0, 1000))

        drone.integrate(engines_speeds, dt)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    ax1.plot(times, Wx)
    ax1.plot(times, Wy)
    ax1.plot(times, Wz)
    ax1.grid()
    ax1.legend(["Wx", "Wy", "Wz"])

    ax2.plot(times, speeds[0])
    ax2.plot(times, speeds[1])
    ax2.plot(times, speeds[2])
    ax2.plot(times, speeds[3])
    ax2.legend(["W1", "W2", "W3", "W4"])
    ax2.grid()

    ax3.plot(times, Yaw)
    ax3.plot(times, Pitch)
    ax3.plot(times, Roll)
    ax3.legend(["Yaw", "Pitch", "Roll"])
    ax3.grid()

    ax4.plot(times, X)
    ax4.plot(times, Y)
    ax4.plot(times, Z)
    ax4.legend(["X", "Y", "Z"])
    ax4.grid()

    plt.show()


if __name__ == "__main__":
    #model_rotation_speeds()
    #model_rotation()
    model_coords()