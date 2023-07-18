import random

from integrator import Integrator


class Differentiator:
    def __init__(self, start_value=0, memory_limit=10):
        self._start_value = start_value
        self._memory = [start_value]
        self._memory_limit = memory_limit

    def differentiate(self, value, dt):
        if len(self._memory) == self._memory_limit:
            self._memory.pop(0)

        self._memory.append(value)

        new_sum = sum(self._memory)/len(self._memory)
        ret = (new_sum - self._start_value) / dt

        self._start_value = new_sum

        return ret


class PID:
    def __init__(self, k_p, k_i, k_d, integrator: Integrator):
        self.__k_p = k_p
        self.__k_i = k_i
        self.__k_d = k_d
        self.__integrator = integrator
        self.__differentiator = Differentiator()

    def reset(self, value):
        self.__integrator.reset(value)

    def calculate(self, value, dt):
        res = self.__k_p * value

        self.__integrator.integrate(value, dt)
        res += self.__k_i * self.__integrator.value

        res += self.__k_d * self.__differentiator.differentiate(value, dt)

        return res


class Filter:
    def __init__(self, memory_limit=10):
        self._memory = []
        self._memory_limit = memory_limit

    def handle(self, value):
        if len(self._memory) == self._memory_limit:
            self._memory.pop(0)

        self._memory.append(value)

        return sum(self._memory) / len(self._memory)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    dt = 0.1
    times = []
    errors = []
    values = []
    filtered = []
    #signal = 5

    #pid = PID(0.1, 0.0, 0.00, integrator=LinearIntegrator())
    diff = Differentiator(memory_limit=10)
    filter = Filter(memory_limit=50)

    for i in range(1000):
        times.append(i*dt)
        errors.append(random.gauss(0, 5))
        filtered.append(filter.handle(errors[i]))
        #values.append(diff.differentiate(errors[i], dt))

    plt.plot(times, errors)
    plt.plot(times, filtered)
    #avg = sum(errors)/ len(errors)
    #plt.plot([times[0], times[-1]], [avg, avg])
    #plt.plot(times, values)
    plt.legend(["Errors", "Values"])
    plt.show()