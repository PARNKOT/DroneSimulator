import typing

import numpy as np


def mix_commands_cross(commands: typing.Sequence):
    thrust = commands[0]
    Wx = commands[1]
    Wy = commands[2]
    Wz = commands[3]

    return np.asfarray([thrust + Wz + Wx - Wy,
                        thrust + Wz - Wx + Wy,
                        thrust - Wz - Wx - Wy,
                        thrust - Wz + Wx + Wy])
