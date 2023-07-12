from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt

from DroneSimulator.src.drone_model import DroneModel
from DroneSimulator.src.command_mixer import mix_commands_cross
from DroneSimulator.src.pid import PID, saturate
from DroneSimulator.src.integrator import IntegratorFactory, LINEAR
from DroneSimulator.src.controls import RotationSpeedsControlLoop, AnglesControlLoop

@dataclass
class DroneParams:
    mass = 0.15
    b_engine = 7e-7
    d_engine = 7e-8
    shoulder = 0.15
    tensor = np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.2, 0.0],
                         [0.0, 0.0, 0.1]], dtype="float64")


drone_params = {
    "mass": 0.15,
    "b_engine": 7e-7,
    "d_engine": 7e-8,
    "shoulder": 0.15,
    "tensor": np.asarray([[0.1, 0.0, 0.0],
                         [0.0, 0.2, 0.0],
                         [0.0, 0.0, 0.1]], dtype="float64"),
}

INTEGRATOR_TYPE = LINEAR


if __name__ == "__main__":
    drone = DroneModel(**drone_params)

    k_p, k_i, k_d = 250, 0.5, 2
    # pidP = PID(1.9, 0.08, 10, IntegratorFactory.create(INTEGRATOR_TYPE))
    pidx = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
    pidy = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))
    pidz = PID(k_p, k_i, k_d, IntegratorFactory.create(INTEGRATOR_TYPE))

    k_p_angles, k_i_angles, k_d_angles = 2.3, 0.1, 2
    pid_yaw = PID(-0.45, -0.01, -0.1, IntegratorFactory.create(INTEGRATOR_TYPE))
    pid_pitch = PID(k_p_angles, k_i_angles, k_d_angles, IntegratorFactory.create(INTEGRATOR_TYPE))
    pid_roll = PID(k_p_angles, k_i_angles, k_d_angles, IntegratorFactory.create(INTEGRATOR_TYPE))

    control1 = RotationSpeedsControlLoop(k_p, k_i, k_d)
    control2 = AnglesControlLoop(k_p_angles, k_i_angles, k_d_angles)

    dt = 0.1 # second

    Pdes = 290
    Wdes_x = 0 * np.pi / 180
    Wdes_y = 0 * np.pi / 180
    Wdes_z = 0 * np.pi / 180

    Yaw_des = 0 * np.pi / 180
    Pitch_des = 10 * np.pi / 180
    Roll_des = 10 * np.pi / 180

    Xdes = 0
    Ydes = 0
    Zdes = 0

    times = []

    Wx = []
    Wy = []
    Wz = []

    Yaw = []
    Pitch = []
    Roll = []

    X, Y, Z = [], [], []

    speeds = [[], [], [], []]

    for i in range(1000):
        times.append(i*dt)

        Wcur_x, Wcur_y, Wcur_z = drone.rotation_speeds
        Xcur, Ycur, Zcur = drone.linear_coords
        Yaw_cur, Pitch_cur, Roll_cur = drone.angles

        Yaw.append(Yaw_cur * 180 / np.pi)
        Pitch.append(Pitch_cur * 180 / np.pi)
        Roll.append(Roll_cur * 180 / np.pi)

        x, y, z = drone.linear_coords
        X.append(x)
        Y.append(y)
        Z.append(z)

        Wx.append(Wcur_x*180/np.pi)
        Wy.append(Wcur_y*180/np.pi)
        Wz.append(Wcur_z*180/np.pi)

        #cmd_thrust = pidP.calculate(Ydes - Ycur, dt)
        # cmd_x = pidx.calculate(Xdes - Xcur, dt)
        # cmd_y = pidy.calculate(Ydes - Ycur, dt)
        # cmd_z = pidz.calculate(Zdes - Zcur, dt)

        cmd_x = pidx.calculate(Wdes_x-Wcur_x, dt)
        cmd_y = pidy.calculate(Wdes_y-Wcur_y, dt)
        cmd_z = pidz.calculate(Wdes_z-Wcur_z, dt)

        cmd_yaw = pid_yaw.calculate(Yaw_des - Yaw_cur, dt)
        cmd_pitch = pid_pitch.calculate(Pitch_des - Pitch_cur, dt)
        cmd_roll = pid_roll.calculate(Roll_des-Roll_cur, dt)

        # speeds_des = [Wdes_x, Wdes_y, Wdes_z]
        thrust, *speeds_des = control2.handle(Pdes, [Yaw_des, Pitch_des, Roll_des],
                                     [Yaw_cur, Pitch_cur, Roll_cur], dt)

        engines_speeds = mix_commands_cross(control1.handle(Pdes, speeds_des,
                                                            [Wcur_x, Wcur_y, Wcur_z], dt))
        # engines_speeds = mix_commands_cross([Pdes, cmd_x, cmd_y, cmd_z])

        for i, speed in enumerate(engines_speeds):
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


    # drone.rotation_angles = np.asfarray([10 * np.pi / 180, 10 * np.pi / 180, 0.0])