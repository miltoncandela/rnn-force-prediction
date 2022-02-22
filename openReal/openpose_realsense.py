import pyrealsense2 as rs
import numpy as np
import cv2
# import openpose
from openpose import pyopenpose as op
from openpose import pyopenpose as op
# import pyopenpose as op
import pyopenpose as op
from pyopenpose import s
# from openpose import pyopenpose as op
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

params = dict()
params["model_folder"] = "D:/openpose/models"  # using git tree
params["face"] = True
params["hand"] = True
params["disable_blending"] = True  # show only keypoints

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Covert image to 8bit/pix
        cv2.convertScaleAbs(depth_image)
        cv2.convertScaleAbs(color_image)

        # Show images
        images = np.hstack((color_image, depth_image))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        k = cv2.waitKey(1)

        # Params
        params = dict()
        params["model_folder"] = "D:/openpose/models"  # using git tree
        params["face"] = True
        params["hand"] = True
        params["disable_blending"] = True # show only keypoints

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        datum.cvInputData = color_image
        opWrapper.emplaceAndPop([datum])
        keypoints = datum.cvOutputData

        coords = depth_image * keypoints  # element-wise
        coords = cv2.convertScaleAbs(coords)  # scale to 8bit/px

        # plot 3D wireframe of the points
        nx , ny = coords.shape
        x = range(nx)
        y = range(ny)
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
        ha.plot_wireframe(X, Y, coords)
        plt.show()

        # Display Image and data for sanity check
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        print("Face keypoints: \n" + str(datum.faceKeypoints))
        print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        cv2.imshow("im", datum.cvOutputData)
        cv2.waitKey(0)

finally:
    # Stop streaming
    pipeline.stop()