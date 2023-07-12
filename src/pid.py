from DroneSimulator.src.integrator import Integrator, LinearIntegrator


class PID:
    def __init__(self, k_p, k_i, k_d, integrator: Integrator):
        self.__k_p = k_p
        self.__k_i = k_i
        self.__k_d = k_d
        self.__past_value = 0
        self.__integrator = integrator

    def calculate(self, value, dt):
        res = self.__k_p * value

        self.__integrator.integrate(value, dt)
        res += self.__k_i * self.__integrator.value

        res += self.__k_d * self.differentiate(value, dt)
        self.__past_value = value

        return res

    def differentiate(self, value, dt):
        return (value - self.__past_value)/dt


def saturate(value, limit):
    if value > limit:
        return limit
    elif value < -limit:
        return -limit
    return value


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    dt = 0.1
    times = [0]
    values = [5]
    signal = 5

    pid = PID(0.1, 0.0, 0.00, integrator=LinearIntegrator())

    for i in range(1000):
        times.append(times[i] + dt)
        values.append(saturate(pid.calculate(signal, dt), 10))

    plt.plot(times, values)
    plt.show()