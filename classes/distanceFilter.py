import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


def state_transition(x, dt):
    # Assuming a constant velocity model for simplicity
    f = np.array([[1, dt],
                  [0, 1]])
    return np.dot(f, x)


def measurement_function(x):
    # Measurement function. In this case, we measure only the distance
    #return np.array([1, 0]).dot(x)
    return np.array([x[0]])


class DistanceFilter(object):
    def __init__(self, dist, frame_rate, alpha=0.1):
        dt = 1 / frame_rate
        points = MerweScaledSigmaPoints(n=2, alpha=alpha, beta=2., kappa=0)
        ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=dt, hx=measurement_function,
                                    fx=state_transition, points=points)

        ukf.x = np.array([dist, 0])
        ukf.P *= 0.2
        ukf.R = np.diag([0.01])
        ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1e-5, block_size=1)

        self.ukf = ukf

    def predict(self):
        self.ukf.predict()

    def update(self, value):
        self.ukf.update(value)

    def get_prediction(self):
        pr
        if self.ukf.log_likelihood < 0.2:
            return 9999

        print(self.ukf.x)
        return self.ukf.x[0] + 3 * self.ukf.x[1]

    def get_frame_prediction(self):
        return np.sum(self.ukf.x)

    def get_current(self):
        return self.ukf.x[0]

# filter = DistanceFilter(dist=1, frame_rate=30)
# filter.predict()
# filter.update(1.2)  # Example measurement update
