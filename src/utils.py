import numpy as np


RPM_TO_RadPS = np.pi/30
Rad_TO_DEGREES = 180/np.pi


def rotate_yaw_matrix(yaw) -> np.ndarray:
    return np.asfarray([[np.cos(yaw), 0, np.sin(yaw)],
                         [0, 1, 0],
                         [-np.sin(yaw), 0, np.cos(yaw)]])


def rotate_pitch_matrix(pitch) -> np.ndarray:
    return np.asfarray([[np.cos(pitch), np.sin(pitch), 0],
                          [-np.sin(pitch), np.cos(pitch), 0],
                          [0, 0, 1]])


def rotate_roll_matrix(roll) -> np.ndarray:
    return np.asfarray([[1, 0, 0],
                         [0, np.cos(roll), np.sin(roll)],
                         [0, -np.sin(roll), np.cos(roll)]])


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
    #pitch = np.arcsin(matrix[1][0])
    pitch = np.arctan2(matrix[1][0], np.sqrt(matrix[1][1]**2 + matrix[1][2]**2))
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

