import numpy as np
import cv2

from classes.pedestrianDetector import PedestrianDetector
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

detector = PedestrianDetector()

while cv2.waitKey(1) < 0:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    height, width, channels = color_image.shape

    objects = detector.detect(color_image)

    print('Found %s objects.' % len(objects))

    depth_image = np.asanyarray(depth_frame.get_data())



    if len(objects) > 0:
        #depth_colormap = objects[0].apply(depth_colormap)
        print(depth_image.shape)
        print(objects[0].mask.shape)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        color_image = objects[0].apply(color_image)

    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    #cv2.imshow('frame', depth_colormap)
    cv2.imshow('frame', color_image)

pipeline.stop()
cv2.destroyAllWindows()
