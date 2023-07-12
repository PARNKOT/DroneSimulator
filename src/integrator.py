from abc import abstractmethod

LINEAR = "linear"


class Integrator:
    def __init__(self, start=0):
        self._value = start

    @property
    def value(self):
        return self._value

    def reset(self, start):
        self._value = start

    @abstractmethod
    def integrate(self, dx, dt):
        pass


class LinearIntegrator(Integrator):
    def integrate(self, dx, dt):
        self._value += dx * dt


class IntegratorFactory:
    @staticmethod
    def create(type: str, start=0) -> Integrator:
        if type.lower() == LINEAR:
            return LinearIntegrator(start)

        return None


def integrate_linear(state, dx, dt):
    return state + dx*dt


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    def dx(t):
        return 0.1*t

    dt = 0.1 # second

    times = [0]
    values = [0]

    integrator = IntegratorFactory.create(LINEAR, values[0])

    for i in range(100):
        times.append(times[i] + dt)
        integrator.integrate(dx(times[i]), dt)
        values.append(integrator.value)

    plt.plot(times, values)
    plt.show()
