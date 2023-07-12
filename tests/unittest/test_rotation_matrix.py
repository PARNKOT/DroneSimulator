from unittest import TestCase, main

import numpy
import numpy as np

from DroneSimulator.src.utils import rotation_matrix


class TestRotationMatrix(TestCase):
    def test_yaw(self):
        yaw, pitch, roll = 90 * np.pi/180, 0, 0
        start_vector = np.asfarray([1, 0, 0])
        end_vector = np.asfarray([0, 0, -1])

        matrix = rotation_matrix(yaw, pitch, roll)
        calc_vector = matrix.dot(start_vector)

        self.assert_(numpy.allclose(end_vector, calc_vector),
                     msg=f"yaw = {yaw}, pitch = {pitch}, roll = {roll};"
                         f"start = {start_vector}, expected = {end_vector}, get = {calc_vector}")

    def test_pitch(self):
        yaw, pitch, roll = 0, 90 * np.pi/180, 0
        start_vector = np.asfarray([1, 0, 0])
        end_vector = np.asfarray([0, -1, 0])

        matrix = rotation_matrix(yaw, pitch, roll)
        calc_vector = matrix.dot(start_vector)

        self.assert_(numpy.allclose(end_vector, calc_vector),
                     msg=f"yaw = {yaw}, pitch = {pitch}, roll = {roll};"
                         f"start = {start_vector}, expected = {end_vector}, get = {calc_vector}")

    def test_roll(self):
        yaw, pitch, roll = 0, 0, 90 * np.pi/180
        start_vector = np.asfarray([1, 0, 0])
        end_vector = np.asfarray([1, 0, 0])

        matrix = rotation_matrix(yaw, pitch, roll)
        calc_vector = matrix.dot(start_vector)
        self.assert_(numpy.allclose(end_vector, calc_vector),
                     msg=f"yaw = {yaw}, pitch = {pitch}, roll = {roll};"
                         f"start = {start_vector}, expected = {end_vector}, get = {calc_vector}")

    def test_all_angles(self):
        yaw, pitch, roll = 45 * np.pi/180, 90 * np.pi/180, 45 * np.pi/180
        start_vector = np.asfarray([1, 0, 0])
        end_vector = np.asfarray([0, -1, 0])

        matrix = rotation_matrix(yaw, pitch, roll)
        calc_vector = matrix.dot(start_vector)

        self.assert_(numpy.allclose(end_vector, calc_vector),
                     msg=f"yaw = {yaw}, pitch = {pitch}, roll = {roll};"
                         f"start = {start_vector}, expected = {end_vector}, get = {calc_vector}")


if __name__ == "__main__":
    main()
