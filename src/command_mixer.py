import typing

import numpy as np


def mix_commands_cross(thrust, speeds: typing.Sequence):
    Wx = speeds[0]
    Wy = speeds[1]
    Wz = speeds[2]

    return np.asfarray([thrust + Wz + Wx - Wy,   # engine 0
                        thrust + Wz - Wx + Wy,   # engine 1
                        thrust - Wz - Wx - Wy,   # engine 2
                        thrust - Wz + Wx + Wy])  # engine 3
