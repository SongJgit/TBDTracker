from numpy.core.multiarray import zeros as zeros
from .botsort_kalman import BotKalman
import numpy as np


class TrackTrackKalman(BotKalman):
    order_by_dim = False
    """Bot Kalman with NSA."""

    def update(self, z, score):
        """update step.

        Args:
            z: observation x-y-w-h format

        K_n = P_{n, n - 1} * H^T * (H P_{n, n - 1} H^T + R)^{-1}
        x_{n, n} = x_{n, n - 1} + K_n * (z - H * x_{n, n - 1})
        P_{n, n} = (I - K_n * H) P_{n, n - 1} (I - K_n * H)^T + K_n R_n
        """

        std = [
            self._std_weight_position * self.kf.x[2], self._std_weight_position * self.kf.x[3],
            self._std_weight_position * self.kf.x[2], self._std_weight_position * self.kf.x[3]]

        std = [(1. - score) * x for x in std]

        R = np.diag(np.square(std))

        self.kf.update(z=z, R=R)
