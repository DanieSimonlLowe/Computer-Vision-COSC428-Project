import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


def state_transition(state, dt):
    f = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  ])
    out = np.dot(f, state)
    return out


def measurement_function(state):
    return state[:2]


class DistanceFilter(object):
    def __init__(self, dist, x, frame_rate, alpha=0.1):
        dt = 1 / frame_rate
        points = MerweScaledSigmaPoints(n=4, alpha=alpha, beta=2., kappa=0)
        ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, hx=measurement_function,
                                    fx=state_transition, points=points)

        ukf.x = np.array([dist, x, 0, 0])
        ukf.P *= 0.2
        ukf.R = np.diag([0.01, 0.01])
        ukf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=1e-5, block_size=1)

        self.ukf = ukf

    def predict(self):
        self.ukf.predict()

    def update(self, dist, x):
        self.ukf.update(np.array([dist, x]))

    def overlap(self):
        # if self.ukf.log_likelihood < 0.2:
        #     return False

        start = complex(self.ukf.x[0], self.ukf.x[1])
        end = complex(self.ukf.x[0] + 3 * self.ukf.x[2], self.ukf.x[1] + 3 * self.ukf.x[3])
        print(start, end)

        if abs(start) < 3 or abs(end) < 3:
            return True

        ab = end - start

        if abs(ab) == 0:
            return False

        ac = -start
        ab_normalized = ab / abs(ab)
        ap_distance = ac.real * ab_normalized.real + ac.imag * ab_normalized.imag
        ap = ap_distance * ab_normalized

        ap_proportion = ap_distance / abs(ab)
        in_segment = 0 <= ap_proportion <= 1

        cp = ap - ac
        in_circle = abs(cp) < 3
        return in_segment and in_circle

    def get_current(self):
        return self.ukf.x[:2]

# filter = DistanceFilter(dist=1, x=1, frame_rate=30)
# filter.predict()
# filter.update(1.2, 1.1)  # Example measurement update
# print(filter.get_prediction())
