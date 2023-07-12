import numpy as np
from unittest import TestCase, main

from DroneSimulator.src.utils import convert_angles_speed


class TestSpeedConvertion(TestCase):
    def test_yaw_rotate(self):
        yaw, pitch, roll = 90 * np.pi/180, 0, 0
        local_speeds = np.asfarray([1, 2, 3]) # wx, wy, wz
        converted_speeds = np.asfarray([-2, 3, 1]) # dpsi, dtetta, dgamma

        calc_speeds = convert_angles_speed([yaw, pitch, roll], local_speeds)

        self.assert_(np.allclose(converted_speeds, calc_speeds), msg=f"expected = {converted_speeds}, get = {calc_speeds}")

    def test_pitch_rotate(self):
        yaw, pitch, roll = 0, 60 * np.pi/180, 0
        local_speeds = np.asfarray([1, 2, 3]) # wx, wy, wz
        converted_speeds = np.asfarray([-1, 3, 1]) # dpsi, dtetta, dgamma

        calc_speeds = convert_angles_speed([yaw, pitch, roll], local_speeds)

        self.assert_(np.allclose(converted_speeds, calc_speeds), msg=f"expected = {converted_speeds}, get = {calc_speeds}")


if __name__ == "__main__":
    main()
