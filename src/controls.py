import math
import random
import typing
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt

from drone_model import DroneMini
from pid import PID, Filter
from integrator import IntegratorFactory, LINEAR
from utils import saturate, saturate_min_max, rotate_yaw_matrix, Rad_TO_DEGREES, RPM_TO_RadPS
from command_mixer import mix_commands_cross
from metrics import DroneMetrics, show_drone_graphics


drone_params = {
    "mass": 1,
    "b_engine": 26.5e-6,
    "d_engine": 0.6e-6,
    "shoulders": [0.15],
    "tensor": np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.1, 0.0],
                         [0.0, 0.0, 0.1]], dtype="float64"),
}

INTEGRATOR_TYPE = LINEAR

metrics = DroneMetrics()

GYRO_ERROR = 1 # degrees per second
LINEAR_SPEED_ERROR = 0.05 # meters per second
COORDS_ERROR = 0.0 # meter


@dataclass
class MeasuredParams:
    Coords = np.asfarray([0, 0, 0])
    Linear_speeds = np.asfarray([0, 0, 0])
    Angles = np.asfarray([0, 0, 0])
    Rotation_speeds = np.asfarray([0, 0, 0])


class RotationSpeedsControlLoop:
    def __init__(self, k_p, k_i, k_d):
        self.__pid_wx = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_wy = PID(k_p + 350, k_i + 10, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_wz = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        limit = 10
        self._filters = [
            Filter(memory_limit=limit),
            Filter(memory_limit=limit),
            Filter(memory_limit=limit),
        ]
        self.limits = None

    def set_initial_state(self, init_speeds: typing.Sequence):
        self.__pid_wx.reset(init_speeds[0])
        self.__pid_wy.reset(init_speeds[1])
        self.__pid_wz.reset(init_speeds[2])

    def handle(self, speeds_des: typing.Sequence, speeds_cur: typing.Sequence, dt):
        wx_des, wy_des, wz_des = speeds_des
        wx_cur, wy_cur, wz_cur = speeds_cur

        if self.limits is not None:
            wx_des = saturate(wx_des, self.limits[0])
            wy_des = saturate(wy_des, self.limits[1])
            wz_des = saturate(wz_des, self.limits[2])

        delta_wx = wx_des-wx_cur
        delta_wy = wy_des-wy_cur
        delta_wz = wz_des-wz_cur

        wx_cmd = self.__pid_wx.calculate(delta_wx, dt)
        wy_cmd = self.__pid_wy.calculate(delta_wy, dt)
        wz_cmd = self.__pid_wz.calculate(delta_wz, dt)

        wx_cmd = self._filters[0].handle(wx_cmd)
        wy_cmd = self._filters[1].handle(wy_cmd)
        wz_cmd = self._filters[2].handle(wz_cmd)

        return np.asfarray([wx_cmd, wy_cmd, wz_cmd])


class AnglesControlLoop:
    def __init__(self, k_p, k_i, k_d):
        self.__pid_yaw = PID(k_p+3, k_i+0.5, k_d+1, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_pitch = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_roll = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.limits = None

    def set_initial_state(self, init_angles: typing.Sequence):
        self.__pid_yaw.reset(init_angles[0])
        self.__pid_pitch.reset(init_angles[1])
        self.__pid_roll.reset(init_angles[2])

    def handle(self, angles_des: typing.Sequence, angles_cur: typing.Sequence, dt):
        yaw_des, pitch_des, roll_des = angles_des
        yaw_cur, pitch_cur, roll_cur = angles_cur

        if self.limits is not None:
            yaw_des = saturate(yaw_des, self.limits[0])
            pitch_des = saturate(pitch_des, self.limits[1])
            roll_des = saturate(roll_des, self.limits[2])

        wx_cmd = self.__pid_roll.calculate(roll_des - roll_cur, dt)
        wy_cmd = self.__pid_yaw.calculate(yaw_des - yaw_cur, dt)
        wz_cmd = self.__pid_pitch.calculate(pitch_des - pitch_cur, dt)

        return np.asfarray([wx_cmd, -wy_cmd, wz_cmd])


class LinearSpeedsControlLoop:
    def __init__(self, k_p, k_i, k_d):
        self.__pid_VN = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_VH = PID(50, 50, 0.0, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_VE = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.limits = None

    def set_initial_state(self, init_state: typing.Sequence):
        self.__pid_VN.reset(init_state[0])
        self.__pid_VH.reset(init_state[1])
        self.__pid_VE.reset(init_state[2])

    def handle(self, speeds_des: typing.Sequence, speeds_cur: typing.Sequence, dt):
        VN_des, VH_des, VE_des = speeds_des
        VN_cur, VH_cur, VE_cur = speeds_cur

        if self.limits is not None:
            VN_des = saturate(VN_des, self.limits[0])
            VH_des = saturate(VH_des, self.limits[1])
            VE_des = saturate(VE_des, self.limits[2])

        VN_cmd = self.__pid_VN.calculate(VN_des - VN_cur, dt)
        VH_cmd = self.__pid_VH.calculate(VH_des - VH_cur, dt)
        VE_cmd = self.__pid_VE.calculate(VE_des - VE_cur, dt)

        return np.asfarray([-VN_cmd, VH_cmd, VE_cmd])


class CoordsControlLoop:
    def __init__(self, k_p, k_i, k_d):
        self.__pid_thrust = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_x = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_z = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        limit = 10
        self._filters = [
            Filter(memory_limit=limit),
            Filter(memory_limit=limit),
            Filter(memory_limit=limit),
        ]

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

        thrust_cmd = self._filters[0].handle(thrust_cmd)
        x_cmd = self._filters[1].handle(x_cmd)
        z_cmd = self._filters[2].handle(z_cmd)

        return np.asfarray([x_cmd, thrust_cmd, z_cmd])


class Controller:
    control1_params = {
        "k_p": 100,
        "k_i": 2.5,
        "k_d": 0,
    }

    control2_params = {
        "k_p": 1.5,
        "k_i": 0.1,
        "k_d": 0,
    }

    control3_params = {
        "k_p": 0.4,
        "k_i": 0.003,
        "k_d": 0.1,
    }

    control4_params = {
        "k_p": 0.5,
        "k_i": 0.001,
        "k_d": 0.0,
    }

    def __init__(self):
        self.__controller_rotation_speeds = RotationSpeedsControlLoop(**self.control1_params)
        self.__controller_angles = AnglesControlLoop(**self.control2_params)
        self.__controller_linear_speeds = LinearSpeedsControlLoop(**self.control3_params)
        self.__controller_coords = CoordsControlLoop(**self.control4_params)

        k_p, k_i, k_d = 1, 0, 0
        self.__engines_pids = [
            PID(k_p, k_i, k_d, IntegratorFactory.create(LINEAR)),
            PID(k_p, k_i, k_d, IntegratorFactory.create(LINEAR)),
            PID(k_p, k_i, k_d, IntegratorFactory.create(LINEAR)),
            PID(k_p, k_i, k_d, IntegratorFactory.create(LINEAR))
        ]

        min_engine_rpms = 1000
        max_engine_rpms = 7000

        self.min_engine_speed = min_engine_rpms * RPM_TO_RadPS  # радиан в секунду
        self.max_engine_speed = max_engine_rpms * RPM_TO_RadPS  # радиан в секунду

    @property
    def angular_speeds_limits(self):
        return self.__controller_rotation_speeds.limits

    @angular_speeds_limits.setter
    def angular_speeds_limits(self, value):
        self.__controller_rotation_speeds.limits = value

    @property
    def angles_limits(self):
        return self.__controller_angles.limits

    @angles_limits.setter
    def angles_limits(self, value):
        self.__controller_angles.limits = value

    @property
    def linear_speeds_limits(self):
        return self.__controller_linear_speeds.limits

    @linear_speeds_limits.setter
    def linear_speeds_limits(self, value):
        self.__controller_linear_speeds.limits = value

    def handle(self, yaw_des, coords_des: typing.Sequence, measurements: MeasuredParams, dt):
        X_des, Y_des, Z_des = coords_des

        metrics.coords_cmd.append([X_des, Y_des, Z_des])

        Wx_cur, Wy_cur, Wz_cur = measurements.Rotation_speeds
        VN_cur, VH_cur, VE_cur = measurements.Linear_speeds
        Yaw_cur, Pitch_cur, Roll_cur = measurements.Angles
        X_cur, Y_cur, Z_cur = measurements.Coords

        # Команды на линейные скорости (VN, VH, VE)
        X_cmd, thrust_cmd, Z_cmd = self.__controller_coords.handle([X_des, Y_des, Z_des], [X_cur, Y_cur, Z_cur], dt)

        metrics.linear_speeds_cmd.append([X_cmd, thrust_cmd, Z_cmd])

        # Команды на углы поворота (крен, тангаж). Pitch_des, Roll_des
        VN_cmd, VH_cmd, VE_cmd = self.__controller_linear_speeds.handle([X_cmd, thrust_cmd, Z_cmd],
                                                                        [VN_cur, VH_cur, VE_cur], dt)

        metrics.angles_cmd.append([VN_cmd, yaw_des, VE_cmd])

        VN_cmd, _, VE_cmd = rotate_yaw_matrix(-measurements.Angles[0]).dot(np.asfarray([VN_cmd, VH_cmd, VE_cmd]))

        speeds_des = self.__controller_angles.handle([yaw_des, VN_cmd, VE_cmd],
                                              [Yaw_cur, Pitch_cur, Roll_cur], dt)

        metrics.angular_speeds_cmd.append(list(speeds_des))

        engines_speeds = mix_commands_cross(VH_cmd, self.__controller_rotation_speeds.handle(speeds_des,
                                                            [Wx_cur, Wy_cur, Wz_cur], dt))

        for i, speed in enumerate(engines_speeds):
            speed = self.__engines_pids[i].calculate(speed, dt)
            engines_speeds[i] = saturate_min_max(speed, self.min_engine_speed, self.max_engine_speed)

        metrics.engines_speeds.append(list(engines_speeds))

        return engines_speeds



def model_rotation_speeds():
    drone = DroneMini(**drone_params)

    k_p, k_i, k_d = 100, 10.5, 0.0

    control1 = RotationSpeedsControlLoop(k_p, k_i, k_d)
    control1.limits = np.asfarray([200, 200, 200]) / Rad_TO_DEGREES

    min_engine_rpms = 1000
    max_engine_rpms = 7000

    min_engine_speed = min_engine_rpms * RPM_TO_RadPS # радиан в секунду
    max_engine_speed = max_engine_rpms * RPM_TO_RadPS # радиан в секунду

    dt = 0.01  # second
    t = int(5/dt)

    Pdes = 305 # 305 - команда, при которой появляется положительная вертикальная скорость
    Wdes_x = 0 / Rad_TO_DEGREES
    Wdes_y = 0 / Rad_TO_DEGREES
    Wdes_z = 205 / Rad_TO_DEGREES

    times = []

    Wx = []
    Wy = []
    Wz = []

    Yaw = []
    Pitch = []
    Roll = []

    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    measurements = MeasuredParams()

    for i in range(t):
        Wdes_x = (10 + (i//50)*10) / Rad_TO_DEGREES

        Wdes_x = saturate(Wdes_x, 200/Rad_TO_DEGREES)

        times.append(i * dt)

        measurements.Rotation_speeds = drone.rotation_speeds
        measurements.Linear_speeds = drone.linear_speeds
        measurements.Angles = drone.angles
        measurements.Coords = drone.linear_coords

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speeds
        Yaw_cur, Pitch_cur, Roll_cur = drone.angles

        error1 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error2 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error3 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        Wcur_x += error1
        Wcur_y += error2
        Wcur_z += error3

        metrics.linear_speeds.append(list(measurements.Linear_speeds))

        Yaw.append(Yaw_cur * Rad_TO_DEGREES)
        Pitch.append(Pitch_cur * Rad_TO_DEGREES)
        Roll.append(Roll_cur * Rad_TO_DEGREES)

        x, y, z = drone.linear_coords
        X.append(x)
        Y.append(y)
        Z.append(z)

        Wx.append(Wcur_x * Rad_TO_DEGREES)
        Wy.append(Wcur_y * Rad_TO_DEGREES)
        Wz.append(Wcur_z * Rad_TO_DEGREES)

        speeds_des = [Wdes_x, Wdes_y, Wdes_z]

        engines_speeds = mix_commands_cross(Pdes, control1.handle(speeds_des, [Wcur_x, Wcur_y, Wcur_z], dt))

        for i, speed in enumerate(engines_speeds):
            engines_speeds[i] = saturate_min_max(speed, min_engine_speed, max_engine_speed)
            speeds[i].append(engines_speeds[i]/RPM_TO_RadPS)

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

    ax4.plot(times, metrics.linear_speeds)
    ax4.legend(["VN", "VH", "VE"])
    ax4.grid()

    plt.show()


def model_rotation():
    drone = DroneMini(**drone_params)

    control1_params = {
        "k_p": 100,
        "k_i": 10.5,
        "k_d": 0,
    }

    k_p, k_i, k_d = 1.5, 0.1, 0

    min_engine_rpms = 1000
    max_engine_rpms = 7000

    min_engine_speed = min_engine_rpms * RPM_TO_RadPS # радиан в секунду
    max_engine_speed = max_engine_rpms * RPM_TO_RadPS # радиан в секунду

    control1 = RotationSpeedsControlLoop(**control1_params)
    control1.limits = np.asfarray([200, 200, 200]) / Rad_TO_DEGREES

    control2 = AnglesControlLoop(k_p, k_i, k_d)
    control2.limits = np.asfarray([math.inf, 30, 30]) / Rad_TO_DEGREES

    dt = 0.01  # second
    t = int(10/dt)

    Pdes = 305
    Yaw_des = 20 / Rad_TO_DEGREES
    Pitch_des = 35 / Rad_TO_DEGREES
    Roll_des = 40 / Rad_TO_DEGREES

    times = []

    Wx = []
    Wy = []
    Wz = []

    Yaw = []
    Pitch = []
    Roll = []

    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    measurements = MeasuredParams()

    for i in range(t):
        #Wdes_x = (10 + (i//50)*10) / Rad_TO_DEGREES

        #Wdes_x = saturate(Wdes_x, 200/Rad_TO_DEGREES)

        times.append(i * dt)

        measurements.Rotation_speeds = drone.rotation_speeds
        measurements.Linear_speeds = drone.linear_speeds
        measurements.Angles = drone.angles
        measurements.Coords = drone.linear_coords

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speeds
        Yaw_cur, Pitch_cur, Roll_cur = drone.angles

        error1 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error2 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error3 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        Wcur_x += error1
        Wcur_y += error2
        Wcur_z += error3

        metrics.linear_speeds.append(list(measurements.Linear_speeds))

        Yaw.append(Yaw_cur * Rad_TO_DEGREES)
        Pitch.append(Pitch_cur * Rad_TO_DEGREES)
        Roll.append(Roll_cur * Rad_TO_DEGREES)

        x, y, z = drone.linear_coords
        X.append(x)
        Y.append(y)
        Z.append(z)

        Wx.append(Wcur_x * Rad_TO_DEGREES)
        Wy.append(Wcur_y * Rad_TO_DEGREES)
        Wz.append(Wcur_z * Rad_TO_DEGREES)

        speeds_des = control2.handle([Yaw_des, Pitch_des, Roll_des], measurements.Angles, dt)

        engines_speeds = mix_commands_cross(Pdes, control1.handle(speeds_des, [Wcur_x, Wcur_y, Wcur_z], dt))

        for i, speed in enumerate(engines_speeds):
            engines_speeds[i] = saturate_min_max(speed, min_engine_speed, max_engine_speed)
            speeds[i].append(engines_speeds[i]/RPM_TO_RadPS)

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

    ax4.plot(times, metrics.linear_speeds)
    ax4.legend(["VN", "VH", "VE"])
    ax4.grid()

    plt.show()


def model_linear_speeds():
    drone = DroneMini(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(0 / Rad_TO_DEGREES, 0*np.pi/180, 0*np.pi/180)

    control1_params = {
        "k_p": 100,
        "k_i": 2.5,
        "k_d": 0,
    }

    control2_params = {
        "k_p": 1.5,
        "k_i": 0.1,
        "k_d": 0,
    }

    k_p, k_i, k_d = 0.4, 0.003, 0.1

    control1 = RotationSpeedsControlLoop(**control1_params)
    control1.limits = np.asfarray([200, 200, 200]) / Rad_TO_DEGREES

    control2 = AnglesControlLoop(k_p, k_i, k_d)
    control2.limits = np.asfarray([math.inf, 30, 30]) / Rad_TO_DEGREES

    control3 = LinearSpeedsControlLoop(k_p, k_i, k_d)
    control3.limits = [15, 6, 15]

    min_engine_rpms = 1000
    max_engine_rpms = 7000

    min_engine_speed = min_engine_rpms * RPM_TO_RadPS # радиан в секунду
    max_engine_speed = max_engine_rpms * RPM_TO_RadPS # радиан в секунду

    dt = 0.01  # second
    t = int(20/dt)

    VN_des = 15
    VH_des = 6
    VE_des = 15
    Yaw_des = 0

    times = []

    Wx, Wy, Wz = [], [], []
    Yaw, Pitch, Roll = [], [], []
    VN, VH, VE = [], [], []
    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    measurements = MeasuredParams()

    for i in range(t):
        # if i == int(0.5*t):
        #     VH_des = 2

        times.append(i * dt)

        measurements.Rotation_speeds = drone.rotation_speeds
        measurements.Linear_speeds = drone.linear_speeds
        measurements.Angles = drone.angles
        measurements.Coords = drone.linear_coords

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speeds
        VN_cur, VH_cur, VE_cur = drone.linear_speeds
        Yaw_cur, Pitch_cur, Roll_cur = drone.angles
        x, y, z = drone.linear_coords

        error1 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error2 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error3 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        Wcur_x += error1
        Wcur_y += error2
        Wcur_z += error3

        error4 = random.gauss(0, LINEAR_SPEED_ERROR)
        error5 = random.gauss(0, LINEAR_SPEED_ERROR)
        error6 = random.gauss(0, LINEAR_SPEED_ERROR)
        VN_cur += error4
        VH_cur += error5
        VE_cur += error6

        Yaw.append(Yaw_cur * 180 / np.pi)
        Pitch.append(Pitch_cur * 180 / np.pi)
        Roll.append(Roll_cur * 180 / np.pi)

        metrics.linear_speeds.append(list(measurements.Linear_speeds))

        # VN.append(VN_cur)
        # VH.append(VH_cur)
        # VE.append(VE_cur)

        X.append(x)
        Y.append(y)
        Z.append(z)

        Wx.append(Wcur_x * 180 / np.pi)
        Wy.append(Wcur_y * 180 / np.pi)
        Wz.append(Wcur_z * 180 / np.pi)

        VN_cmd, VH_cmd, VE_cmd = control3.handle([VN_des, VH_des, VE_des], [VN_cur, VH_cur, VE_cur], dt)

        VN_cmd, _, VE_cmd = rotate_yaw_matrix(-measurements.Angles[0]).dot(np.asfarray([VN_cmd, VH_cmd, VE_cmd]))

        # Команды тангажа и крена ограничиваются +-30 градусами
        #VN_cmd = saturate(VN_cmd, 30 * np.pi/180)
        #VE_cmd = saturate(VE_cmd, 30 * np.pi / 180)
        #VH_cmd = 0


        speeds_des = control2.handle([Yaw_des, VN_cmd, VE_cmd],
                                     [Yaw_cur, Pitch_cur, Roll_cur], dt)



        engines_speeds = mix_commands_cross(VH_cmd, control1.handle(speeds_des,
                                                            [Wcur_x, Wcur_y, Wcur_z], dt))

        for i, speed in enumerate(engines_speeds):
            engines_speeds[i] = saturate_min_max(speed, min_engine_speed, max_engine_speed)
            speeds[i].append(engines_speeds[i] / RPM_TO_RadPS)

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

    ax4.plot(times, metrics.linear_speeds)
    ax4.legend(["VN", "VH", "VE"])
    ax4.grid()

    plt.show()


def model_coords():
    drone = DroneMini(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(50 / Rad_TO_DEGREES, 0*np.pi/180, 0*np.pi/180)

    control1_params = {
        "k_p": 100,
        "k_i": 2.5,
        "k_d": 0,
    }

    control2_params = {
        "k_p": 1.5,
        "k_i": 0.1,
        "k_d": 0,
    }

    control3_params = {
        "k_p": 0.4,
        "k_i": 0.003,
        "k_d": 0.1,
    }

    control4_params = {
        "k_p": 0.5,
        "k_i": 0.001,
        "k_d": 0.0,
    }

    # k_p, k_i, k_d = 1.0, 0.001, 0.05
    k_p, k_i, k_d = 0.5, 0.001, 0.01

    control1 = RotationSpeedsControlLoop(**control1_params)
    control1.limits = np.asfarray([200, 200, 200]) / Rad_TO_DEGREES

    control2 = AnglesControlLoop(**control2_params)
    control2.limits = np.asfarray([math.inf, 30, 30]) / Rad_TO_DEGREES

    control3 = LinearSpeedsControlLoop(**control3_params)
    control3.limits = [15, 6, 15]

    control4 = CoordsControlLoop(**control4_params)

    min_engine_rpms = 1000
    max_engine_rpms = 7000

    min_engine_speed = min_engine_rpms * RPM_TO_RadPS # радиан в секунду
    max_engine_speed = max_engine_rpms * RPM_TO_RadPS # радиан в секунду

    dt = 0.01  # second
    t = int(60/dt)

    X_des = 200
    Y_des = 50
    Z_des = 300
    Yaw_des = 0 / Rad_TO_DEGREES

    times = []

    Wx, Wy, Wz = [], [], []
    Yaw, Pitch, Roll = [], [], []
    VN, VH, VE = [], [], []
    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    measurements = MeasuredParams()

    for i in range(t):
        # if i == int(0.5*t):
        #     VH_des = 2
        metrics.times.append(i*dt)
        times.append(i * dt)

        measurements.Rotation_speeds = drone.rotation_speeds
        measurements.Linear_speeds = drone.linear_speeds
        measurements.Angles = drone.angles
        measurements.Coords = drone.linear_coords

        metrics.angular_speeds.append([measurements.Rotation_speeds[0] * 180 / np.pi,
                                       measurements.Rotation_speeds[1] * 180 / np.pi,
                                       measurements.Rotation_speeds[2] * 180 / np.pi])

        metrics.angles.append([measurements.Angles[0] * 180 / np.pi,
                               measurements.Angles[1] * 180 / np.pi,
                               measurements.Angles[2] * 180 / np.pi])

        metrics.linear_speeds.append(list(measurements.Linear_speeds))
        metrics.coords.append(list(measurements.Coords))

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speeds
        VN_cur, VH_cur, VE_cur = drone.linear_speeds
        Yaw_cur, Pitch_cur, Roll_cur = drone.angles
        X_cur, Y_cur, Z_cur = drone.linear_coords

        error1 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error2 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        error3 = random.gauss(0, GYRO_ERROR/Rad_TO_DEGREES)
        Wcur_x += error1
        Wcur_y += error2
        Wcur_z += error3

        error4 = random.gauss(0, LINEAR_SPEED_ERROR)
        error5 = random.gauss(0, LINEAR_SPEED_ERROR)
        error6 = random.gauss(0, LINEAR_SPEED_ERROR)
        VN_cur += error4
        VH_cur += error5
        VE_cur += error6

        Yaw.append(Yaw_cur * 180 / np.pi)
        Pitch.append(Pitch_cur * 180 / np.pi)
        Roll.append(Roll_cur * 180 / np.pi)

        Wx.append(Wcur_x * 180 / np.pi)
        Wy.append(Wcur_y * 180 / np.pi)
        Wz.append(Wcur_z * 180 / np.pi)

        X_cmd, thrust_cmd, Z_cmd = control4.handle([X_des, Y_des, Z_des], [X_cur, Y_cur, Z_cur], dt)

        VN_cmd, VH_cmd, VE_cmd = control3.handle([X_cmd, thrust_cmd, Z_cmd], [VN_cur, VH_cur, VE_cur], dt)

        VN_cmd, _, VE_cmd = rotate_yaw_matrix(-measurements.Angles[0]).dot(np.asfarray([VN_cmd, VH_cmd, VE_cmd]))

        # Команды тангажа и крена ограничиваются +-30 градусами
        #VN_cmd = saturate(VN_cmd, 30 * np.pi/180)
        #VE_cmd = saturate(VE_cmd, 30 * np.pi / 180)
        #VH_cmd = 0


        speeds_des = control2.handle([Yaw_des, VN_cmd, VE_cmd],
                                     [Yaw_cur, Pitch_cur, Roll_cur], dt)



        engines_speeds = mix_commands_cross(VH_cmd, control1.handle(speeds_des,
                                                            [Wcur_x, Wcur_y, Wcur_z], dt))

        for i, speed in enumerate(engines_speeds):
            engines_speeds[i] = saturate_min_max(speed, min_engine_speed, max_engine_speed)

        metrics.engines_speeds.append(list(engines_speeds / RPM_TO_RadPS))

        drone.integrate(engines_speeds, dt)

    show_drone_graphics(metrics)


def main():
    drone = DroneMini(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(0 / Rad_TO_DEGREES, 0 / Rad_TO_DEGREES, 0 / Rad_TO_DEGREES)

    controller = Controller()

    controller.angular_speeds_limits = np.asfarray([200, 200, 200]) / Rad_TO_DEGREES
    controller.angles_limits = np.asfarray([math.inf, 30, 30]) / Rad_TO_DEGREES
    controller.linear_speeds_limits = [15, 6, 15]

    dt = 0.01  # second
    t = int(60 / dt)

    X_des = 200
    Y_des = 100
    Z_des = 300
    Yaw_des = 40 * np.pi / 180

    def trajectory(t):
        if t > 30:
            return (t, 200, 500)
        return (t, 200, 0.5*t**2)

    for i in range(t):
        metrics.times.append(i * dt)
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

        error7 = random.gauss(0, COORDS_ERROR)
        error8 = random.gauss(0, COORDS_ERROR)
        error9 = random.gauss(0, COORDS_ERROR)
        measurements.Coords[0] += error7
        measurements.Coords[1] += error8
        measurements.Coords[2] += error9

        metrics.angular_speeds.append([measurements.Rotation_speeds[0] * 180 / np.pi,
                                       measurements.Rotation_speeds[1] * 180 / np.pi,
                                       measurements.Rotation_speeds[2] * 180 / np.pi])

        metrics.angles.append([measurements.Angles[0] * 180 / np.pi,
                               measurements.Angles[1] * 180 / np.pi,
                               measurements.Angles[2] * 180 / np.pi])

        metrics.linear_speeds.append(list(measurements.Linear_speeds))
        metrics.coords.append(list(measurements.Coords))

        #X_des, Y_des, Z_des = trajectory(metrics.times[i])

        engines_speeds = controller.handle(Yaw_des, [X_des, Y_des, Z_des], measurements, dt)

        drone.integrate(engines_speeds, dt)

    show_drone_graphics(metrics)


if __name__ == "__main__":
    #model_rotation_speeds()
    #model_rotation()
    #model_linear_speeds()
    #model_coords()
    main()
