import numpy as np
import cv2

from classes.detectedObect import DetectionState
from classes.pedestrianDetector import PedestrianDetector
import pyrealsense2 as rs

config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
# config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_device_from_file("video.bag")

pipeline = rs.pipeline()
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

detector = PedestrianDetector()

align = rs.align(rs.stream.color)

while cv2.waitKey(1) < 0:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    height, width, channels = color_image.shape

    objects = detector.detect(color_image)

    print('Found %s objects.' % len(objects))

    depth_image = np.asanyarray(depth_frame.get_data())

    for obj in objects:
        state = obj.to_close(depth_image, depth_scale)
        if state == DetectionState.SAFE:
            color_image = obj.highlight(color_image, (0, 255, 0))
        elif state == DetectionState.WARNING:
            color_image = obj.highlight(color_image, (0, 165, 255))
        else:
            color_image = obj.highlight(color_image, (0, 0, 255))


    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow('frame', color_image)

pipeline.stop()
cv2.destroyAllWindows()
