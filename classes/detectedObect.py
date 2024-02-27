from enum import Enum

import numpy as np
import cv2

alpha = 0.5


class DetectionState(Enum):
    SAFE = 1
    WARNING = 2
    DANGER = 3


class DetectedObject:
    def __init__(self, mask):
        self.mask = mask

    def apply(self, image):
        return np.where(self.mask, image, 0)

    def distance(self, image, scale):
        mat = np.where(self.mask, image, np.nan)
        return np.nanmean(mat) * scale

    def to_close(self, image, scale):
        dist = self.distance(image, scale)

        if dist > 5:
            return DetectionState.SAFE
        elif dist > 2:
            return DetectionState.WARNING
        else:
            return DetectionState.DANGER

    def highlight(self, image, color):
        colored_mask = np.expand_dims(self.mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)

        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        return image_combined
