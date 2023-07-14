import pickle
import typing
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt

from sender import Sender
from drone_model import DroneModel
from pid import PID
from integrator import IntegratorFactory, LINEAR
from utils import saturate
from command_mixer import mix_commands_cross


drone_params = {
    "mass": 0.8,
    "b_engine": 26.5e-6,
    "d_engine": 0.6e-6,
    "shoulder": 0.15,
    "tensor": np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.1, 0.0],
                         [0.0, 0.0, 0.1]], dtype="float64"),
}

control1_params = {
    "k_p": 90,
    "k_i": 0.3,
    "k_d": 1,
}

control2_params = {
    "k_p_angles": 5.5,
    "k_i_angles": 0,
    "k_d_angles": 1,
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

INTEGRATOR_TYPE = LINEAR


class RotationSpeedsControlLoop:
    k_p, k_i, k_d = 250, 0.5, 2

    def __init__(self, k_p, k_i, k_d):
        self.__pid_wx = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_wy = PID(k_p+300, k_i+10, k_d+1, IntegratorFactory.create(INTEGRATOR_TYPE))
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
    k_p_yaw, k_i_yaw, k_d_yaw = 3, 0., 0.6

    def __init__(self, k_p_angles, k_i_angles, k_d_angles):
        self.__pid_yaw = PID(3, 0., 0.6, IntegratorFactory.create(INTEGRATOR_TYPE))
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
        wy_cmd = self.__pid_yaw.calculate(yaw_des - yaw_cur, dt)
        wz_cmd = self.__pid_pitch.calculate(pitch_des - pitch_cur, dt)

        return np.asfarray([thrust_cmd, wx_cmd, -wy_cmd, wz_cmd])


class LinearSpeedsControlLoop:
    def __init__(self, k_p, k_i, k_d):
        #15, 0.15, 70
        self.__pid_VN = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
        self.__pid_VH = PID(30, 15, 0, IntegratorFactory.create(INTEGRATOR_TYPE)) # 5, 3.5, 0
        self.__pid_VE = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))

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
        "k_p": 90,
        "k_i": 0.3,
        "k_d": 1,
    }

    control2_params = {
        "k_p_angles": 5.5,
        "k_i_angles": 0,
        "k_d_angles": 1,
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

        Wx_cur, Wy_cur, Wz_cur = measurements.Rotation_speeds
        VN_cur, VH_cur, VE_cur = measurements.Linear_speeds
        Yaw_cur, Pitch_cur, Roll_cur = measurements.Angles
        X_cur, Y_cur, Z_cur = measurements.Coords

        X_cmd, thrust_cmd, Z_cmd = self.__controller_coords.handle([x_des, y_des, z_des], [X_cur, Y_cur, Z_cur], dt)

        VN_cmd, VH_cmd, VE_cmd = self.__controller_linear_speeds.handle([X_cmd, thrust_cmd, Z_cmd], [VN_cur, VH_cur, VE_cur], dt)

        _, VH_cmd, _ = rotation_matrix.dot(np.asfarray([VN_cmd, VH_cmd, VE_cmd]))

        # Команды тангажа и крена ограничиваются +-30 градусами
        VN_cmd = saturate(VN_cmd, self.angles_limits[0])
        VE_cmd = saturate(VE_cmd, self.angles_limits[1])

        thrust, *speeds_des = self.__controller_angles.handle(VH_cmd, [yaw_des, VN_cmd, VE_cmd],
                                              [Yaw_cur, Pitch_cur, Roll_cur], dt)

        engines_speeds = mix_commands_cross(self.__controller_rotation_speeds.handle(thrust, speeds_des,
                                                            [Wx_cur, Wy_cur, Wz_cur], dt))

        return engines_speeds


def model_rotation_speeds():
    from matplotlib import pyplot as plt
    from src.drone_model import DroneModel
    from src.command_mixer import mix_commands_cross
    from src.utils import saturate, saturate_min_max
    drone = DroneModel(**drone_params)

    k_p, k_i, k_d = 90, 0.3, 1
    #k_p_angles, k_i_angles, k_d_angles = 2.3, 0.1, 2

    control1 = RotationSpeedsControlLoop(k_p, k_i, k_d)
    #control2 = AnglesControlLoop(k_p_angles, k_i_angles, k_d_angles)

    dt = 0.01  # second
    t = int(1/dt)

    Pdes = 290
    Wdes_x = 10 * np.pi / 180
    Wdes_y = 5 * np.pi / 180
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

    for i in range(t):
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
    from src.drone_model import DroneModel
    from src.command_mixer import mix_commands_cross
    from src.utils import saturate
    drone = DroneModel(**drone_params)

    k_p_angles, k_i_angles, k_d_angles = 5.5, 0, 1

    control1 = RotationSpeedsControlLoop(**control1_params)
    control2 = AnglesControlLoop(k_p_angles, k_i_angles, k_d_angles)

    dt = 0.01  # second
    t = int(10/dt)

    Pdes = 290
    Yaw_des = 5 * np.pi / 180
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

    for i in range(t):
        if i == int(0.5 * t):
            Pitch_des = 0 * np.pi / 180

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

        for i, speed in enumerate(drone.engines_speeds):
            speeds[i].append(speed)

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
    # ax = plt.axes(projection="3d")
    # ax.plot3D(X,Y,Z)
    # ax.grid()
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    plt.show()


def model_linear_speeds():
    from matplotlib import pyplot as plt
    from src.drone_model import DroneModel
    from src.command_mixer import mix_commands_cross
    from src.utils import saturate, saturate_min_max
    drone = DroneModel(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(0, 10*np.pi/180, -15*np.pi/180)

    k_p, k_i, k_d = 0.3, 0.005, 0.0

    control1 = RotationSpeedsControlLoop(**control1_params)
    control2 = AnglesControlLoop(**control2_params)
    control3 = LinearSpeedsControlLoop(k_p, k_i, k_d)

    dt = 0.01  # second
    t = int(20/dt)

    VN_des = 5
    VH_des = 3
    VE_des = 10
    Yaw_des = 0

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
        x, y, z = drone.linear_coords

        Yaw.append(Yaw_cur * 180 / np.pi)
        Pitch.append(Pitch_cur * 180 / np.pi)
        Roll.append(Roll_cur * 180 / np.pi)

        VN.append(VN_cur)
        VH.append(VH_cur)
        VE.append(VE_cur)

        X.append(x)
        Y.append(y)
        Z.append(z)

        Wx.append(Wcur_x * 180 / np.pi)
        Wy.append(Wcur_y * 180 / np.pi)
        Wz.append(Wcur_z * 180 / np.pi)

        VN_cmd, VH_cmd, VE_cmd = control3.handle([VN_des, VH_des, VE_des], [VN_cur, VH_cur, VE_cur], dt)

        _, VH_cmd, _ = drone.rotation_matrix.dot(np.asfarray([VN_cmd, VH_cmd, VE_cmd]))

        #VH_cmd *= VH_cmd/VH_cmd_new
        #VH_cmd += 2

        # Команды тангажа и крена ограничиваются +-30 градусами
        VN_cmd = saturate(VN_cmd, 30 * np.pi/180)
        VE_cmd = saturate(VE_cmd, 30 * np.pi / 180)
        #VH_cmd = 0


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


if __name__ == "__main__":
    #sender = Sender("localhost", 10100)

    drone = DroneModel(**drone_params)
    drone.linear_speeds = np.asfarray([0, 0, 0])
    drone.linear_coords = np.asfarray([0, 0, 0])
    drone.set_init_angles(0, 0 * np.pi / 180, 0)

    controller = Controller()

    dt = 0.01  # second
    t = int(30 / dt)

    X_des = 0#500
    Y_des = 0#200
    Z_des = 0#1000
    Yaw_des = 30*np.pi/180

    times = []

    Wx, Wy, Wz = [], [], []
    Yaw, Pitch, Roll = [], [], []
    VN, VH, VE = [], [], []
    X, Y, Z = [], [], []

    speeds = [[], [], [], []]


    def trajectory(t):
        if t > 30:
            return (t, 200, 500)
        return (t, 200, 0.5*t**2)


    for i in range(t):
        times.append(i * dt)
        measurements = MeasuredParams()

        measurements.Rotation_speeds = drone.rotation_speeds
        measurements.Linear_speeds = drone.linear_speeds
        measurements.Angles = drone.angles
        measurements.Coords = drone.linear_coords

        state = np.asfarray([measurements.Coords, measurements.Angles])

        #sender.put(pickle.dumps(state))

        Wx.append(measurements.Rotation_speeds[0] * 180 / np.pi)
        Wy.append(measurements.Rotation_speeds[1] * 180 / np.pi)
        Wz.append(measurements.Rotation_speeds[2] * 180 / np.pi)

        Yaw.append(measurements.Angles[0] * 180 / np.pi)
        Pitch.append(measurements.Angles[1] * 180 / np.pi)
        Roll.append(measurements.Angles[2] * 180 / np.pi)

        VN.append(measurements.Linear_speeds[0])
        VH.append(measurements.Linear_speeds[1])
        VE.append(measurements.Linear_speeds[2])

        X.append(measurements.Coords[0])
        Y.append(measurements.Coords[1])
        Z.append(measurements.Coords[2])

        #X_des, Y_des, Z_des = trajectory(times[-1])

        engines_speeds = controller.handle(Yaw_des, [X_des, Y_des, Z_des], measurements, drone.rotation_matrix, dt)

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

    # ax = plt.axes(projection="3d")
    # ax.plot3D(X,Y,Z)
    # ax.grid()
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    plt.show()
