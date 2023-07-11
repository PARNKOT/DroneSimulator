from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt

from drone_model import DroneModel
from command_mixer import mix_commands_cross
from pid import PID, saturate
from integrator import IntegratorFactory, LINEAR


@dataclass
class DroneParams:
    mass = 0.15
    b_engine = 7e-7
    d_engine = 7e-7
    shoulder = 0.15
    tensor = np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.2, 0.0],
                         [0.0, 0.0, 0.1]], dtype="float64")


drone_params = {
    "mass": 0.15,
    "b_engine": 7e-7,
    "d_engine": 7e-7,
    "shoulder": 0.15,
    "tensor": np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.2, 0.0],
                         [0.0, 0.0, 0.1]], dtype="float64"),
}


if __name__ == "__main__":
    drone = DroneModel(**drone_params)

    k_p, k_i, k_d = 0.1, 1, 0
    pidP = PID(1.9, 0.08, 10, IntegratorFactory.create(LINEAR))
    pidx = PID(k_p, k_i, k_d, IntegratorFactory.create(LINEAR))
    pidy = PID(k_p, k_i, k_d, IntegratorFactory.create(LINEAR))
    pidz = PID(k_p, k_i, k_d, IntegratorFactory.create(LINEAR))

    dt = 0.1 # second

    Pdes = 290
    Wdes_x = 0 * np.pi / 180
    Wdes_y = 0
    Wdes_z = 0

    Xdes = 0
    Ydes = 200
    Zdes = 0

    times = []
    Wx = []
    Wy = []
    Wz = []

    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    for i in range(1000):
        times.append(i*dt)

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speed_state
        Xcur, Ycur, Zcur = drone.linear_state

        Wx.append(Wcur_x*180/np.pi)
        Wy.append(Wcur_y*180/np.pi)
        Wz.append(Wcur_z*180/np.pi)

        cmd_thrust = pidP.calculate(Ydes - Ycur, dt)
        cmd_x = pidx.calculate(Xdes - Xcur, dt)
        cmd_y = pidy.calculate(Ydes - Ycur, dt)
        cmd_z = pidz.calculate(Zdes - Zcur, dt)

        # cmd_x = pidx.calculate(Wdes_x-Wcur_x, dt)
        # cmd_y = pidy.calculate(Wdes_y-Wcur_y, dt)
        # cmd_z = pidz.calculate(Wdes_z-Wcur_z, dt)

        engines_speeds = mix_commands_cross([cmd_thrust, 0, 0, 0])

        for i, speed in enumerate(engines_speeds):
            engines_speeds[i] = saturate(speed, 2500)
            speeds[i].append(saturate(speed, 2500))

        x, y, z = drone.linear_state
        X.append(x)
        Y.append(y)
        Z.append(z)

        drone.integrate(engines_speeds, dt)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

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

    ax3.plot(times, X)
    ax3.plot(times, Y)
    ax3.plot(times, Z)
    ax3.legend(["X", "Y", "Z"])
    ax3.grid()


    plt.show()


    # drone.rotation_angles = np.asfarray([10 * np.pi / 180, 10 * np.pi / 180, 0.0])