from matplotlib import pyplot as plt


class Metrics:
    def __init__(self):
        self.metrics = {}

    def set_metric_key(self, key: str):
        #self.metrics[key] = []
        setattr(self, key, list())

    def add_metric_value(self, key, value):
        getattr(self, key).append(value)

    def get_metrics(self, key: str):
        return getattr(self, key)


class DroneMetrics(Metrics):
    def __init__(self):
        super().__init__()
        self.times = []
        self.angular_speeds = []
        self.angular_speeds_cmd = []
        self.angles = []
        self.angles_cmd = []
        self.linear_speeds = []
        self.linear_speeds_cmd = []
        self.coords = []
        self.coords_cmd = []
        self.angles = []
        self.engines_speeds = []


def show_drone_graphics(metrics: DroneMetrics):
    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3)

    ax11.plot(metrics.times, metrics.angular_speeds)
    ax11.plot(metrics.times, metrics.angular_speeds_cmd, ls="--", lw=2.0)
    ax11.grid()
    ax11.legend(["Wx", "Wy", "Wz", "Wx_cmd", "Wy_cmd", "Wz_cmd"])

    ax12.plot(metrics.times, metrics.engines_speeds)
    ax12.legend(["W1", "W2", "W3", "W4"])
    ax12.grid()

    ax13.plot(metrics.times, metrics.angles)
    ax13.plot(metrics.times, metrics.angles_cmd, ls="--", lw=2.0)
    ax13.legend(["Yaw", "Pitch", "Roll", "Yaw_cmd", "Pitch_cmd", "Roll_cmd"])
    ax13.grid()

    ax21.plot(metrics.times, metrics.linear_speeds)
    ax21.plot(metrics.times, metrics.linear_speeds_cmd, ls="--", lw=2.0)
    ax21.legend(["VN", "VH", "VE", "VN_cmd", "VH_cmd", "VE_cmd"])
    ax21.grid()

    ax22.plot(metrics.times, metrics.coords)
    ax22.plot(metrics.times, metrics.coords_cmd, ls="--", lw=2.0)
    ax22.legend(["X", "Y", "Z", "X_cmd", "Y_cmd", "Z_cmd"])
    ax22.grid()

    plt.show()
