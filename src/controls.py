import math
import typing
import numpy as np
from dataclasses import dataclass

from drone_model import DroneModel, DroneMini
from pid import PID
from integrator import IntegratorFactory, LINEAR
from utils import saturate, rotate_yaw_matrix, Rad_TO_DEGREES, RPM_TO_RadPS
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


class RotationSpeedsControlLoop:
    k_p, k_i, k_d = 250, 0.5, 2

    def __init__(self, k_p, k_i, k_d):
        self.__pid_wx = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_wy = PID(k_p + 350, k_i + 10, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_wz = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
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

        wx_cmd = self.__pid_wx.calculate(wx_des-wx_cur, dt)
        wy_cmd = self.__pid_wy.calculate(wy_des-wy_cur, dt)
        wz_cmd = self.__pid_wz.calculate(wz_des-wz_cur, dt)

        return np.asfarray([wx_cmd, wy_cmd, wz_cmd])


class AnglesControlLoop:
    k_p_angles, k_i_angles, k_d_angles = 2.3, 0.1, 2
    k_p_yaw, k_i_yaw, k_d_yaw = 3, 0., 0.6

    def __init__(self, k_p, k_i, k_d):
        #self.__pid_yaw = PID(3, 0., 0.6, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_yaw = PID(k_p+10, k_i+1, k_d+1, IntegratorFactory.create(INTEGRATOR_TYPE))
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
        #15, 0.15, 70
        self.__pid_VN = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_VH = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE)) # 5, 3.5, 0
        self.__pid_VE = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.limits = None

    def set_initial_state(self, init_state: typing.Sequence):
        self.__pid_VN.reset(init_state[0])
        self.__pid_VH.reset(init_state[1])
        self.__pid_VE.reset(init_state[2])

    def handle(self, speeds_des: typing.Sequence, speeds_cur: typing.Sequence, dt):
        VN_des, VH_des, VE_des = speeds_des
        VN_cur, VH_cur, VE_cur = speeds_cur

        VN_cmd = self.__pid_VN.calculate(VN_des - VN_cur, dt)
        VH_cmd = self.__pid_VH.calculate(VH_des - VH_cur, dt)
        VE_cmd = self.__pid_VE.calculate(VE_des - VE_cur, dt)

        if self.limits is not None:
            VN_cmd = saturate(VN_cmd, self.limits[0])
            VH_cmd = saturate(VH_cmd, self.limits[1])
            VE_cmd = saturate(VE_cmd, self.limits[2])

        return np.asfarray([-VN_cmd, VH_cmd, VE_cmd])


class CoordsControlLoop:
    def __init__(self, k_p, k_i, k_d):
        #15, 0.15, 70
        self.__pid_thrust = PID(0.3, 0.0, 0, IntegratorFactory.create(INTEGRATOR_TYPE))
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


@dataclass
class MeasuredParams:
    Coords = np.asfarray([0, 0, 0])
    Linear_speeds = np.asfarray([0, 0, 0])
    Angles = np.asfarray([0, 0, 0])
    Rotation_speeds = np.asfarray([0, 0, 0])


class Controller:
    control1_params = {
        "k_p": 100,
        "k_i": 1.5,
        "k_d": 0,
    }

    control2_params = {
        "k_p": 5,
        "k_i": 0.1,
        "k_d": 1,
    }

    control3_params = {
        "k_p": 0.3,
        "k_i": 0.005,
        "k_d": 0,
    }

    control4_params = {
        "k_p": 0.08,
        "k_i": 0.0001,
        "k_d": 0,
    }

    def __init__(self):
        self.__controller_rotation_speeds = RotationSpeedsControlLoop(**self.control1_params)
        self.__controller_angles = AnglesControlLoop(**self.control2_params)
        self.__controller_linear_speeds = LinearSpeedsControlLoop(**self.control3_params)
        self.__controller_coords = CoordsControlLoop(**self.control4_params)

        self.angles_limits = (30 * np.pi/180, 30*np.pi/180)

    def handle(self, yaw_des, coords_des: typing.Sequence, measurements: MeasuredParams, rotation_matrix: np.ndarray, dt):
        x_des, y_des, z_des = coords_des

        metrics.coords_cmd.append([x_des, y_des, z_des])

        Wx_cur, Wy_cur, Wz_cur = measurements.Rotation_speeds
        VN_cur, VH_cur, VE_cur = measurements.Linear_speeds
        Yaw_cur, Pitch_cur, Roll_cur = measurements.Angles
        X_cur, Y_cur, Z_cur = measurements.Coords

        # Команды на линейные скорости (VN, VH, VE)
        X_cmd, thrust_cmd, Z_cmd = self.__controller_coords.handle([x_des, y_des, z_des], [X_cur, Y_cur, Z_cur], dt)

        #_, thrust_cmd, _ = rotation_matrix.dot(np.asfarray([X_cmd, thrust_cmd, Z_cmd]))

        metrics.linear_speeds_cmd.append([X_cmd, thrust_cmd, Z_cmd])

        # Команды на углы поворота (крен, тангаж). Pitch_des, Roll_des
        VN_cmd, VH_cmd, VE_cmd = self.__controller_linear_speeds.handle([X_cmd, thrust_cmd, Z_cmd], [VN_cur, VH_cur, VE_cur], dt)

        metrics.angles_cmd.append([VN_cmd, yaw_des, VE_cmd])

        #_, VH_cmd_new, _ = rotation_matrix.dot(np.asfarray([VN_cmd, VH_cmd, VE_cmd]))
        VN_cmd, _, VE_cmd = rotate_yaw_matrix(-measurements.Angles[0]).dot(np.asfarray([VN_cmd, VH_cmd, VE_cmd]))

        # Команды тангажа и крена ограничиваются +-30 градусами
        VN_cmd = saturate(VN_cmd, self.angles_limits[0])
        VE_cmd = saturate(VE_cmd, self.angles_limits[1])

        thrust, *speeds_des = self.__controller_angles.handle(VH_cmd, [yaw_des, VN_cmd, VE_cmd],
                                              [Yaw_cur, Pitch_cur, Roll_cur], dt)

        metrics.angular_speeds_cmd.append(list(speeds_des))

        engines_speeds = mix_commands_cross(self.__controller_rotation_speeds.handle(thrust, speeds_des,
                                                            [Wx_cur, Wy_cur, Wz_cur], dt))

        metrics.engines_speeds.append(list(engines_speeds))

        return engines_speeds



def model_rotation_speeds():
    from matplotlib import pyplot as plt
    from src.drone_model import DroneMini
    from src.command_mixer import mix_commands_cross
    from src.utils import saturate, saturate_min_max

    drone = DroneMini(**drone_params)

    k_p, k_i, k_d = 100, 1.5, 0

    control1 = RotationSpeedsControlLoop(k_p, k_i, k_d)
    control1.limits = np.asfarray([200, 200, 200]) / Rad_TO_DEGREES

    min_engine_rpms = 1000
    max_engine_rpms = 7000

    min_engine_speed = min_engine_rpms * RPM_TO_RadPS # радиан в секунду
    max_engine_speed = max_engine_rpms * RPM_TO_RadPS # радиан в секунду

    dt = 0.01  # second
    t = int(1/dt)

    Pdes = 305 # 305 - команда, при которой появляется положительная вертикальная скорость
    Wdes_x = 10 / Rad_TO_DEGREES
    Wdes_y = 1 / Rad_TO_DEGREES
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
    from matplotlib import pyplot as plt
    from src.drone_model import DroneModel, DroneMini
    from src.command_mixer import mix_commands_cross
    from src.utils import saturate, saturate_min_max

    drone = DroneMini(**drone_params)

    control1_params = {
        "k_p": 100,
        "k_i": 1.5,
        "k_d": 0,
    }

    k_p, k_i, k_d = 5, 0.1, 1

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
    Yaw_des = 30 / Rad_TO_DEGREES
    Pitch_des = 5 / Rad_TO_DEGREES
    Roll_des = 5 / Rad_TO_DEGREES

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
    from matplotlib import pyplot as plt
    from src.drone_model import DroneModel
    from src.command_mixer import mix_commands_cross
    from src.utils import saturate, saturate_min_max

    drone = DroneMini(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(0, 0*np.pi/180, 0*np.pi/180)

    control1_params = {
        "k_p": 100,
        "k_i": 1.5,
        "k_d": 0,
    }

    control2_params = {
        "k_p": 5,
        "k_i": 0.1,
        "k_d": 1,
    }

    k_p, k_i, k_d = 1000, 0.0, 0.0

    control1 = RotationSpeedsControlLoop(**control1_params)
    control2 = AnglesControlLoop(**control2_params)
    control3 = LinearSpeedsControlLoop(k_p, k_i, k_d)
    control3.limits = [15, 6, 15]

    min_engine_rpms = 1000
    max_engine_rpms = 7000

    min_engine_speed = min_engine_rpms * RPM_TO_RadPS # радиан в секунду
    max_engine_speed = max_engine_rpms * RPM_TO_RadPS # радиан в секунду

    dt = 0.01  # second
    t = int(10/dt)

    VN_des = 0
    VH_des = 3
    VE_des = 0
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
    from matplotlib import pyplot as plt
    from src.drone_model import DroneModel
    from src.command_mixer import mix_commands_cross
    from src.utils import saturate, saturate_min_max
    drone = DroneModel(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(0, 0 * np.pi / 180, 0)

    k_p, k_i, k_d = 0.08, 0.0001, 0.0

    control1 = RotationSpeedsControlLoop(**control1_params)
    control2 = AnglesControlLoop(**control2_params)
    control3 = LinearSpeedsControlLoop(**control3_params)
    control4 = CoordsControlLoop(k_p, k_i, k_d)

    dt = 0.01  # second
    t = int(50 / dt)

    X_des = 100
    Y_des = 100
    Z_des = 100
    Yaw_des = 30*np.pi/180

    times = []

    Wx, Wy, Wz = [], [], []
    Yaw, Pitch, Roll = [], [], []
    VN, VH, VE = [], [], []
    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    for i in range(t):
        # if i == int(0.5*t):
        #     VH_des = 2

        times.append(i * dt)

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speeds
        VN_cur, VH_cur, VE_cur = drone.linear_speeds
        Yaw_cur, Pitch_cur, Roll_cur = drone.angles
        X_cur, Y_cur, Z_cur = drone.linear_coords

        Yaw.append(Yaw_cur * 180 / np.pi)
        Pitch.append(Pitch_cur * 180 / np.pi)
        Roll.append(Roll_cur * 180 / np.pi)

        VN.append(VN_cur)
        VH.append(VH_cur)
        VE.append(VE_cur)

        X.append(X_cur)
        Y.append(Y_cur)
        Z.append(Z_cur)

        Wx.append(Wcur_x * 180 / np.pi)
        Wy.append(Wcur_y * 180 / np.pi)
        Wz.append(Wcur_z * 180 / np.pi)

        x_cmd, thrust_cmd, z_cmd = control4.handle([X_des, Y_des, Z_des], [X_cur, Y_cur, Z_cur], dt)

        VN_cmd, VH_cmd, VE_cmd = control3.handle([x_cmd, thrust_cmd, z_cmd], [VN_cur, VH_cur, VE_cur], dt)

        _, VH_cmd, _ = drone.rotation_matrix.dot(np.asfarray([VN_cmd, VH_cmd, VE_cmd]))

        # VH_cmd *= VH_cmd/VH_cmd_new
        # VH_cmd += 2

        # Команды тангажа и крена ограничиваются +-30 градусами
        VN_cmd = saturate(VN_cmd, 30 * np.pi / 180)
        VE_cmd = saturate(VE_cmd, 30 * np.pi / 180)
        # VH_cmd = 0

        thrust, *speeds_des = control2.handle(VH_cmd, [Yaw_des, VN_cmd, VE_cmd],
                                              [Yaw_cur, Pitch_cur, Roll_cur], dt)

        # for i, speed in enumerate(speeds_des):
        #     speeds_des[i] = saturate(speed, 40)

        engines_speeds = mix_commands_cross(control1.handle(thrust, speeds_des,
                                                            [Wcur_x, Wcur_y, Wcur_z], dt))

        for i, speed in enumerate(drone.engines_speeds):
            speeds[i].append(speed)

        drone.integrate(engines_speeds, dt)

    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3)

    ax11.plot(times, Wx)
    ax11.plot(times, Wy)
    ax11.plot(times, Wz)
    ax11.grid()
    ax11.legend(["Wx", "Wy", "Wz"])

    ax12.plot(times, speeds[0])
    ax12.plot(times, speeds[1])
    ax12.plot(times, speeds[2])
    ax12.plot(times, speeds[3])
    ax12.legend(["W1", "W2", "W3", "W4"])
    ax12.grid()

    ax13.plot(times, Yaw)
    ax13.plot(times, Pitch)
    ax13.plot(times, Roll)
    ax13.legend(["Yaw", "Pitch", "Roll"])
    ax13.grid()

    ax21.plot(times, VN)
    ax21.plot(times, VH)
    ax21.plot(times, VE)
    ax21.legend(["VN", "VH", "VE"])
    ax21.grid()

    ax22.plot(times, X)
    ax22.plot(times, Y)
    ax22.plot(times, Z)
    ax22.legend(["X", "Y", "Z"])
    ax22.grid()

    plt.show()


def main():
    drone = DroneModel(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(0 * np.pi / 180, 0 * np.pi / 180, 0 * np.pi / 180)

    controller = Controller()
    #metrics = DroneMetrics()

    dt = 0.01  # second
    t = int(50 / dt)

    X_des = 300
    Y_des = 50
    Z_des = 300
    Yaw_des = 0 * np.pi / 180

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

        metrics.angular_speeds.append([measurements.Rotation_speeds[0] * 180 / np.pi,
                                       measurements.Rotation_speeds[1] * 180 / np.pi,
                                       measurements.Rotation_speeds[2] * 180 / np.pi])

        metrics.angles.append([measurements.Angles[0] * 180 / np.pi,
                               measurements.Angles[1] * 180 / np.pi,
                               measurements.Angles[2] * 180 / np.pi])

        metrics.linear_speeds.append(list(measurements.Linear_speeds))
        metrics.coords.append(list(measurements.Coords))

        #X_des, Y_des, Z_des = trajectory(times[-1])

        engines_speeds = controller.handle(Yaw_des, [X_des, Y_des, Z_des], measurements, drone.rotation_matrix, dt)

        drone.integrate(engines_speeds, dt)

    show_drone_graphics(metrics)


if __name__ == "__main__":
    #model_rotation_speeds()
    model_rotation()
    #model_linear_speeds()
    #main()
