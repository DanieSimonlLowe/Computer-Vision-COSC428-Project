import numpy as np
import cv2

alpha = 0.5


class DetectedObject:
    def __init__(self, mask):
        self.mask = mask

    def apply(self, image):
        return np.where(self.mask, image, 0)
