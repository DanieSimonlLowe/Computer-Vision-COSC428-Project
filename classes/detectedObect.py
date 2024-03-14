from enum import Enum

import numpy as np
import cv2

alpha = 0.5
assumed_arm_length = 0.5  # based on 35cm average male arm span with a bit added for wigle room.

min_area = 100

kernel = np.ones((3, 3), np.uint8)


class DetectionState(Enum):
    SAFE = 1
    WARNING = 2
    DANGER = 3


class DetectedObject:
    def __init__(self, mask):
        self.mask = mask
        self._distance = None
        self._contour = None

    """
    calculates the distance of an object from the camera.
    :arg image: the depth image that is used to calculate the distance to the object
    :arg scale: used to scale the values got from the depth camera to get them in meters
    :return distance: the distance from the camera in meters
    :raise error: is thrown if the image or scale has not been provided and the distance is not calculated yet
    """

    def distance(self, image=None, scale=None):
        if self._distance is not None:
            return self._distance
        if image is None or scale is None:
            raise Exception("distance has not been calculated yet.")

        # gets the median distance of the image which is in the mask.
        mat = np.where(self.mask, image, np.nan)
        dist = np.nanmedian(mat) * scale

        # Converted the image into the correct format is needed so that the edge detection and contours can run
        # scales the image so that 7.5 meters is 255
        scaled_image = image / (7.5 / (255 * scale))
        # apply the mask to the image and sets all values outside the mask as 255
        regions_image = np.where(self.mask, scaled_image, 255).astype(np.uint8)

        dist_min = dist
        for level in range(5, 250, 5):
            thresh = cv2.inRange(regions_image, level, level + 5)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=4)

            contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            has_changed = False

            for cnt in contours:
                # gets the area of the contour and checks that it is large enough
                area = cv2.contourArea(cnt)
                if area > min_area:
                    # creates a mask of everything inside the contour
                    mask_cnt = np.zeros_like(mat)
                    mask_cnt = cv2.drawContours(mask_cnt, cnt, -1, 255, -1)

                    # gets the median distance of the things inside the contour
                    masked_cnt = np.where(mask_cnt, image, np.nan)
                    dist_cnt = np.nanmedian(masked_cnt) * scale

                    # if this dist is less then the min dist record the dist and the contour
                    if dist_cnt < dist_min:
                        dist_min = dist_cnt
                        self._contour = cnt
                        has_changed = True

            if has_changed:
                break

        # returns the min dist if it makes sense else returns the median dist.
        if dist - dist_min < assumed_arm_length:
            self._distance = dist_min
            return dist_min
        else:
            self._distance = dist
            return dist

    """
    gets the warning state based upon the distance to the camera
    """

    def get_warning_state(self, image, scale):
        dist = self.distance(image, scale)

        if dist > 5:
            return DetectionState.SAFE
        elif dist > 3:
            return DetectionState.WARNING
        else:
            return DetectionState.DANGER

    """
    highlight the image with the provided color on the mask and outlines the section that is closest to the camera if one is detected
    """

    def highlight(self, image, color):
        colored_mask = np.expand_dims(self.mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)

        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        if self._contour is not None:
            image_combined = cv2.drawContours(image_combined, self._contour, -1, (0, 0, 0), 3)

        return image_combined

    """
    shows the distance from the camera on the image.
    """

    def show_distance(self, image):
        image = cv2.putText(image, str(self._distance), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2, cv2.LINE_AA)
        return image
