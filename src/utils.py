import numpy as np


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


def get_angles_from_rotation_matrix(matrix: np.ndarray):
    yaw = np.arctan2(matrix[2][0], matrix[0][0])
    pitch = np.arcsin(matrix[1][0])
    roll = np.arctan2(-matrix[1][2], matrix[1][1])

    return np.asfarray([yaw, pitch, roll])

def saturate(value, limit):
    if value > limit:
        return limit
    elif value < -limit:
        return -limit
    return value

def saturate_min_max(value, min_limit, max_limit):
    if value > max_limit:
        return max_limit
    elif value < min_limit:
        return min_limit
    return value

# def convert_angles_speed(angles: typing.Sequence, angles_speed_from: np.ndarray):
#     yaw, pitch, roll = angles[0], angles[1], angles[2]
#     convert_matrix = np.asfarray([[0, -np.cos(roll)/np.cos(pitch), np.sin(roll)/np.cos(pitch)],
#                                   [0, np.sin(roll), np.cos(roll)],
#                                   [1, -np.tan(pitch)*np.cos(roll), np.tan(pitch)*np.sin(roll)]])
#
#     return convert_matrix.dot(angles_speed_from)