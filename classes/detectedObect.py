from enum import Enum

import numpy as np
import cv2

alpha = 0.5
assumed_arm_length = 0.5  # based on 35 with a bit added for wigle room.

min_area = 50


class DetectionState(Enum):
    SAFE = 1
    WARNING = 2
    DANGER = 3


class DetectedObject:
    def __init__(self, mask):
        self.mask = mask
        self._distance = None
        self._contour = None

    def apply(self, image):
        return np.where(self.mask, image, 0)

    def distance(self, image, scale):
        if self._distance is not None:
            return self._distance

        mat = np.where(self.mask, image, np.nan)
        dist = np.nanmedian(mat) * scale

        mat_cont = np.where(self.mask, image, 255)
        mat_cont = mat_cont.astype(np.uint8)
        edges = cv2.Canny(mat_cont, 75, 300)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dist_min = dist
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > min_area:
                mask_cnt = np.zeros_like(mat)
                mask_cnt = cv2.drawContours(mask_cnt, cnt, -1, 255, -1)

                masked_cnt = np.where(mask_cnt, image, np.nan)
                dist_cnt = np.nanmedian(masked_cnt) * scale
                if dist_cnt < dist_min:
                    dist_min = dist_cnt
                    self._contour = cnt

        if dist - dist_min < 0.5:
            self._distance = dist_min
            return dist_min
        else:
            self._distance = dist
            return dist

    def to_close(self, image, scale):
        dist = self.distance(image, scale)

        if dist > 5:  # 5
            return DetectionState.SAFE
        elif dist > 3:  # 3
            return DetectionState.WARNING
        else:
            return DetectionState.DANGER

    def highlight(self, image, color):
        colored_mask = np.expand_dims(self.mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)

        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        if self._contour is not None:
            print('draw Contour')
            image_combined = cv2.drawContours(image_combined, self._contour, -1, (0, 0, 0), 3)

        return image_combined

    def show_distance(self, image):
        image = cv2.putText(image, str(self._distance), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2, cv2.LINE_AA)
        return image
