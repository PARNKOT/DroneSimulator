import numpy as np
import numpy.linalg
from integrator import integrate_linear


def rotation_matrix(yaw, pitch, roll) -> np.ndarray:
    """
    @param yaw: курс, [рад]
    @param pitch: тангаж, [рад]
    @param roll: крен, [рад]
    @return: матрица поворота из нормальной земной СК в связанную СК
    """

    R_roll = np.asfarray([[1, 0, 0],
                         [0, np.cos(roll), np.sin(roll)],
                         [0, -np.sin(roll), np.cos(roll)]])

    R_pitch = np.asfarray([[np.cos(pitch), np.sin(pitch), 0],
                          [-np.sin(pitch), np.cos(pitch), 0],
                          [0, 0, 1]])

    R_yaw = np.asfarray([[np.cos(yaw), 0, np.sin(yaw)],
                         [0, 1, 0],
                         [-np.sin(yaw), 0, np.cos(yaw)]])

    return R_roll.dot(R_pitch.dot(R_yaw))

# def rotation_matrix(yaw, pitch, roll) -> np.ndarray:
#     """
#     @param yaw: курс, [рад]
#     @param pitch: тангаж, [рад]
#     @param roll: крен, [рад]
#     @return: матрица поворота из нормальной земной СК в связанную СК
#     """
#
#     R_roll = np.asfarray([[1, 0, 0],
#                          [0, np.cos(roll), np.sin(roll)],
#                          [0, -np.sin(roll), np.cos(roll)]])
#
#     R_yaw = np.asfarray([[np.cos(yaw), np.sin(yaw), 0],
#                           [-np.sin(yaw), np.cos(yaw), 0],
#                           [0, 0, 1]])
#
#     R_pitch = np.asfarray([[np.cos(pitch), 0, np.sin(pitch)],
#                          [0, 1, 0],
#                          [-np.sin(pitch), 0, np.cos(pitch)]])
#
#     return R_roll.dot(R_pitch.dot(R_yaw))


# angles = [psi, tetta, gamma]
# def convert_angles_speed(angles: np.ndarray, angles_speed_from: np.ndarray):
#     yaw, pitch, roll = angles[0], angles[1], angles[2]
#     convert_matrix = np.asfarray([[0, np.sin(roll)/np.cos(pitch), -np.cos(roll)/np.cos(pitch)],
#                                   [1, np.cos(roll), np.sin(roll)],
#                                   [1, np.tan(pitch)*np.sin(roll), -np.tan(pitch)*np.cos(roll)]])
#
#     return convert_matrix.dot(angles_speed_from)

def convert_angles_speed(angles: np.ndarray, angles_speed_from: np.ndarray):
    yaw, pitch, roll = angles[0], angles[1], angles[2]
    convert_matrix = np.asfarray([[0, -np.cos(roll)/np.cos(pitch), np.sin(roll)/np.cos(pitch)],
                                  [1, np.sin(roll), np.cos(roll)],
                                  [1, -np.tan(pitch)*np.cos(roll), np.tan(pitch)*np.sin(roll)]])

    return convert_matrix.dot(angles_speed_from)

class DroneModel:
    def __init__(self, mass: float, tensor: np.ndarray, shoulder: float, integrate_func=integrate_linear):
        self.mass = mass
        self.tensor = tensor
        self.b_engine = 0
        self.d_engine = 0

        # Стоит подумать над созданием отдельного класса конфигурации дрона (квадро-, октокоптер и т.д.)
        self.shoulder = shoulder

        self.linear_speed_state = np.asfarray([0, 0, 0])
        self.linear_state = np.asfarray([0, 0, 0]) # Xc, Yc, Zc

        self.rotation_speed_state = np.asfarray([0, 0, 0]) # Wx, Wy, Wz
        self.rotation_state = np.asfarray([0, 0, 0])

        self.rotation_angles = np.asfarray([0, 0, 0])

        self.integrate_func = integrate_func

    def calculate_right_linear(self, engines_speed: np.ndarray, angles: np.ndarray) -> np.ndarray:
        res: np.ndarray
        G = np.asfarray([0, -self.mass*9.81, 0])

        res = np.asfarray([0, self.b_engine*np.sum(engines_speed**2), 0])
        matrix = rotation_matrix(*angles)
        res = matrix.T.dot(res) + G

        return res

    def active_moments(self, engines_speed: np.ndarray) -> np.ndarray:
        k_engine = self.shoulder*self.b_engine
        config_matrix = np.asarray([[0.0, -k_engine, 0.0, k_engine],
                                    [-self.d_engine, self.d_engine, -self.d_engine, self.d_engine],
                                    [k_engine, 0.0, -k_engine, 0.0]], dtype="float64")
        return np.dot(config_matrix, engines_speed**2)

    def calculate_rights_rotation(self, engines_speed: np.ndarray) -> np.ndarray:
        res: np.ndarray

        inverse_tensor = numpy.linalg.inv(self.tensor)

        res = self.active_moments(engines_speed) - np.cross(self.rotation_speed_state,
                                                            self.tensor.dot(self.rotation_speed_state))

        return inverse_tensor.dot(res)

    def integrate_linear(self, engines_speed: np.ndarray, dt):
        angles = self.rotation_angles
        self.linear_speed_state = self.integrate_func(self.linear_speed_state,
                                                      self.calculate_right_linear(engines_speed, angles), dt)

        self.linear_state = self.integrate_func(self.linear_state, self.linear_speed_state, dt)

    def integrate_rotation(self, engines_speed: np.ndarray, dt):
        self.rotation_speed_state = self.integrate_func(self.rotation_speed_state,
                                                      self.calculate_rights_rotation(engines_speed), dt)

        self.rotation_state = self.integrate_func(self.rotation_state, self.rotation_speed_state, dt)

    def integrate(self, engines_speed: np.ndarray, dt):
        self.rotation_angles = self.integrate_func(self.rotation_angles,
                            convert_angles_speed(self.rotation_angles, self.rotation_speed_state), dt)
        self.integrate_linear(engines_speed, dt)
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

    model.rotation_angles = np.asfarray([0.0, 20*np.pi/180, 0.0])

    engines_speed = np.asarray([2500, 2500, 2500, 2500], dtype="float64")

    dt = 0.1
    times = [0]
    Wx, Wy, Wz = [0.0], [0.0], [0.0]
    Angle1 = [0.0]
    Angle2 = [0.0]
    Angle3 = [0.0]
    Yaw = [0.0]
    Pitch = [20]
    Roll = [0.0]
    Xc = [0.0]
    Yc = [0.0]
    Zc = [0.0]
    X = [0.0]
    Y = [0.0]
    Z = [0.0]

    for i in range(100):
        times.append(times[i]+dt)
        model.integrate(engines_speed, dt)
        Wx.append(model.rotation_speed_state[0])
        Wy.append(model.rotation_speed_state[1])
        Wz.append(model.rotation_speed_state[2])
        Angle1.append(model.rotation_state[0])
        Angle2.append(model.rotation_state[1])
        Angle3.append(model.rotation_state[2])
        Yaw.append(model.rotation_angles[0] * (180/np.pi))
        Pitch.append(model.rotation_angles[1] * (180/np.pi))
        Roll.append(model.rotation_angles[2] * (180/np.pi))
        Xc.append(model.linear_state[0])
        Yc.append(model.linear_state[1])
        Zc.append(model.linear_state[2])

    # plt.plot(times, Wx)
    # plt.plot(times, Wy)
    # plt.plot(times, Wz)
    # plt.legend(["X", "Y", "Z"])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax2.plot(times, Yaw)
    ax2.plot(times, Pitch)
    ax2.plot(times, Roll)
    ax2.legend(["Yaw", "Pitch", "Roll"])

    ax1.plot(times, Xc)
    ax1.plot(times, Yc)
    ax1.plot(times, Zc)
    ax1.legend(["X", "Y", "Z"])
    plt.grid()
    plt.show()

