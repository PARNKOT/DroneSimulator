import numpy as np
import numpy.linalg
from DroneSimulator.src.integrator import integrate_linear
from DroneSimulator.src.utils import rotation_matrix, get_angles_from_rotation_matrix, saturate, saturate_min_max


class DroneModel:
    def __init__(self, mass: float, tensor: np.ndarray, shoulder: float, integrate_func=integrate_linear,
                 b_engine=0, d_engine=0):
        self.mass = mass
        self.tensor = tensor
        self.b_engine = b_engine
        self.d_engine = d_engine
        self.limit = 1000

        # Стоит подумать над созданием отдельного класса конфигурации дрона (квадро-, октокоптер и т.д.)
        self.shoulder = shoulder

        self.linear_speeds = np.asfarray([0, 0, 0]) # VN (Север), VH (Высота), VE (Восток)
        self.linear_coords = np.asfarray([0, 0, 0]) # XN (Север), YH (Высота), ZE (Восток)

        self.rotation_speeds = np.asfarray([0, 0, 0]) # Wx, Wy, Wz
        self.angles = np.asfarray([0, 0, 0]) # Yaw, Pitch, Roll

        self.rotation_matrix = rotation_matrix(self.angles[0], self.angles[1], self.angles[2]).T

        self.integrate_func = integrate_func

    def set_init_angles(self, yaw, pitch, roll):
        self.angles = np.asfarray([yaw, pitch, roll])
        self.rotation_matrix = rotation_matrix(yaw, pitch, roll).T

    def calculate_rights_linear(self, engines_speed: np.ndarray) -> np.ndarray:
        res: np.ndarray
        G = np.asfarray([0, -self.mass*9.81, 0])

        res = np.asfarray([0, self.b_engine*np.sum(engines_speed**2)/self.mass, 0])
        return self.rotation_matrix.dot(res) + G

    def rotation_matrix_rights(self):
        wx, wy, wz = self.rotation_speeds
        speeds_matrix = np.asfarray([[0, -wz, wy],
                                     [wz, 0, -wx],
                                     [-wy, wx, 0]])
        return self.rotation_matrix.dot(speeds_matrix)

    # def active_moments(self, engines_speed: np.ndarray) -> np.ndarray:
    #     k_engine = self.shoulder*self.b_engine
    #     config_matrix = np.asarray([[0.0, -k_engine, 0.0, k_engine],
    #                                 [-self.d_engine, self.d_engine, -self.d_engine, self.d_engine],
    #                                 [k_engine, 0.0, -k_engine, 0.0]], dtype="float64")
    #     return np.dot(config_matrix, engines_speed**2)

    def active_moments(self, engines_speed: np.ndarray) -> np.ndarray:
        k_engine = self.shoulder*self.b_engine
        config_matrix = np.asfarray([[k_engine, -k_engine, -k_engine, k_engine],
                                    [-self.d_engine, self.d_engine, -self.d_engine, self.d_engine],
                                    [k_engine, k_engine, -k_engine, -k_engine]])
        return np.dot(config_matrix, engines_speed**2)

    def calculate_rights_rotation(self, engines_speed: np.ndarray) -> np.ndarray:
        res: np.ndarray

        inverse_tensor = numpy.linalg.inv(self.tensor)

        res = self.active_moments(engines_speed) - np.cross(self.rotation_speeds,
                                                            self.tensor.dot(self.rotation_speeds))

        return inverse_tensor.dot(res)

    def integrate_linear(self, engines_speed: np.ndarray, dt):
        self.linear_coords = self.integrate_func(self.linear_coords, self.linear_speeds, dt)

        self.linear_speeds = self.integrate_func(self.linear_speeds,
                                                      self.calculate_rights_linear(engines_speed), dt)

    def integrate_rotation(self, engines_speed: np.ndarray, dt):
        self.rotation_speeds = self.integrate_func(self.rotation_speeds,
                                                      self.calculate_rights_rotation(engines_speed), dt)

    def integrate_rotation_matrix(self, dt):
        self.rotation_matrix = self.integrate_func(self.rotation_matrix, self.rotation_matrix_rights(), dt)
        self.angles = get_angles_from_rotation_matrix(self.rotation_matrix)

    def integrate(self, engines_speed: np.ndarray, dt):
        for i, speed in enumerate(engines_speed):
            engines_speed[i] = saturate_min_max(speed, 0, self.limit)

        self.integrate_linear(engines_speed, dt)
        self.linear_coords[1] = saturate_min_max(self.linear_coords[1], 0, self.linear_coords[1])
        self.integrate_rotation_matrix(dt)
        self.integrate_rotation(engines_speed, dt)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mass = 0.5
    tensor = np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.2, 0.0],
                         [0.0, 0.0, 0.3]], dtype="float64")
    shoulder = 0.15

    model = DroneModel(0.5, tensor, shoulder)
    model.b_engine = 7e-7
    model.d_engine = 7e-7

    model.set_init_angles(0*np.pi/180, 0*np.pi/180, 0*np.pi/180)

    engines_speed = np.asfarray([2505, 2500, 2500, 2505])

    dt = 0.1
    times = []
    Wx, Wy, Wz = [0.0], [0.0], [0.0]
    Angle1 = [0.0]
    Angle2 = [0.0]
    Angle3 = [0.0]
    Yaw = []
    Pitch = []
    Roll = []
    XN = []
    YH = []
    ZE = []


    for i in range(201):
        times.append(i*dt)
        if i == 70:
            engines_speed = np.asfarray([2500, 2500, 2500, 2500])

        model.integrate(engines_speed, dt)
        yaw, pitch, roll = get_angles_from_rotation_matrix(model.rotation_matrix)
        Yaw.append(yaw * 180 / np.pi)
        Pitch.append(pitch * 180 / np.pi)
        Roll.append(roll * 180 / np.pi)
        # Wx.append(model.rotation_speed_state[0])
        # Wy.append(model.rotation_speed_state[1])
        # Wz.append(model.rotation_speed_state[2])
        # Angle1.append(model.rotation_state[0])
        # Angle2.append(model.rotation_state[1])
        # Angle3.append(model.rotation_state[2])
        # Yaw.append(model.rotation_angles[0] * (180/np.pi))
        # Pitch.append(model.rotation_angles[1] * (180/np.pi))
        # Roll.append(model.rotation_angles[2] * (180/np.pi))
        XN.append(model.linear_coords[0])
        YH.append(model.linear_coords[1])
        ZE.append(model.linear_coords[2])

    # plt.plot(times, Yaw)
    # plt.plot(times, Pitch)
    # plt.plot(times, Roll)
    # plt.legend(["Yaw", "Pitch", "Roll"])
    # plt.grid()
    # plt.show()

    # plt.plot(times, Wx)
    # plt.plot(times, Wy)
    # plt.plot(times, Wz)
    # plt.legend(["X", "Y", "Z"])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax2.plot(times, Yaw)
    ax2.plot(times, Pitch)
    ax2.plot(times, Roll)
    ax2.legend(["Yaw", "Pitch", "Roll"])

    ax1.plot(times, XN)
    ax1.plot(times, YH)
    ax1.plot(times, ZE)
    ax1.legend(["X", "Y", "Z"])
    plt.grid()
    plt.show()

