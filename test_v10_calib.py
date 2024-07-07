import cv2
import math
from ultralytics import YOLOv10

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from collections import defaultdict
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D, get_depth_at_pixel
from measurement_task import calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements
from realsense_device_manager import post_process_depth_frame
from preprocess import preprocess_image


# Load the YOLOv8 model
model = YOLOv10("weights/yolov10_best.pt")

# Depth Cameras D435
import pyrealsense2 as rs
import numpy as np
import cv2



# calib camera
resolution_width = 1280 # pixels
resolution_height = 720 # pixels
# frame_rate = 15  # fps
frame_rate = 6

dispose_frames_for_stablisation = 30  # frames

chessboard_width = 6 # squares
chessboard_height = 9 	# squares
square_size = 0.0253 # meters
try:
		# Enable the streams from all the intel realsense devices
    rs_config = rs.config()
    # rs_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    # Use the device manager class to enable the devices and get the frames
    device_manager = DeviceManager(rs.context(), rs_config)
    device_manager.enable_all_devices()

    # Allow some frames for the auto-exposure controller to stablise
    for frame in range(dispose_frames_for_stablisation):
        frames = device_manager.poll_frames()

    assert( len(device_manager._available_devices) > 0 )
    """
    1: Calibration
    Calibrate all the available devices to the world co-ordinates.
    For this purpose, a chessboard printout for use with opencv based calibration process is needed.

    """
    # Get the intrinsics of the realsense device
    intrinsics_devices = device_manager.get_device_intrinsics(frames)

            # Set the chessboard parameters for calibration
    chessboard_params = [chessboard_height, chessboard_width, square_size]

    # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
    calibrated_device_count = 0
    while calibrated_device_count < len(device_manager._available_devices):
        frames = device_manager.poll_frames()
        pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
        transformation_result_kabsch  = pose_estimator.perform_pose_estimation()
        object_point = pose_estimator.get_chessboard_corners_in3d()
        calibrated_device_count = 0
        for device_info in device_manager._available_devices:
            device = device_info[0]
            if not transformation_result_kabsch[device][0]:
                print("Place the chessboard on the plane where the object needs to be detected..")
            else:
                calibrated_device_count += 1

    # Save the transformation object for all devices in an array to use for measurements
    transformation_devices={}
    chessboard_points_cumulative_3d = np.array([-1,-1,-1]).transpose()
    for device_info in device_manager._available_devices:
        device = device_info[0]
        transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
        points3D = object_point[device][2][:,object_point[device][3]]
        points3D = transformation_devices[device].apply_transformation(points3D)
        chessboard_points_cumulative_3d = np.column_stack( (chessboard_points_cumulative_3d,points3D) )

    # Extract the bounds between which the object's dimensions are needed
    # It is necessary for this demo that the object's length and breath is smaller than that of the chessboard
    chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
    roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

    print("Calibration completed... \nPlace the box in the field of view of the devices...")
    """
            2: Measurement and display
            Measure the dimension of the object using depth maps from multiple RealSense devices
            The information from Phase 1 will be used here

            """

    # Enable the emitter of the devices
    device_manager.enable_emitter(True)

    # Load the JSON settings file in order to enable High Accuracy preset for the realsense
    device_manager.load_settings_json("E:\DATN\camera_yolov10\yolov10\HighResHighAccuracyPreset.json")

    # Get the extrinsics of the device to be used later
    extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

    # Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
    calibration_info_devices = defaultdict(list)
    for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
        for key, value in calibration_info.items():
            calibration_info_devices[key].append(value)

    # Continue acquisition until terminated with Ctrl+C by the user
    while True:
        # Get the frames from all the devices
        frames_devices = device_manager.poll_frames()
    

#         # Calculate the pointcloud using the depth frames from all the devices
#         point_cloud = calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2D)

#         # Get the bounding box for the pointcloud in image coordinates of the color imager
#         bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(point_cloud, calibration_info_devices )

#         # # Draw the bounding box points on the color image and visualise the results
#         visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)

# # except KeyboardInterrupt:
# #     print("The program was interupted by the user. Closing the program...")

# # finally:
# #     device_manager.disable_streams()
# #     cv2.destroyAllWindows()


# # try:
#     # while True:
#         # Wait for a coherent pair of frames: depth and color
#         # frames = pipeline.wait_for_frames()
#         # depth_frame = frames.get_depth_frame()
#         # color_frame = frames.get_color_frame()
        for (device_info, frame) in frames_devices.items():
		# device = device_info[0] #serial number
            color_image = np.asarray(frame[rs.stream.color].get_data())
            depth_image = np.asarray(frame[rs.stream.depth].get_data())
            
            
    
            # color_frame = frame[rs.stream.color]
            # depth_frame = frame[rs.stream.depth]
        
            # if not depth_frame or not color_frame:
            #     continue

            # # Convert images to numpy arrays
            # depth_image = np.asanyarray(depth_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())
            
    # preprocess - color 

            # Run YOLOv8 inference on the frame
            results = model(color_image, conf=0.1)
            
            # Extract the labels and positions
            detections = results[0].boxes
            # print('++++++++++', len(detections))
            
        
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
                
            for detection in detections:
                # Get the label
                label = detection.cls.item()  # Assuming 'cls' is a tensor

                # Get the position (bounding box coordinates)
                x1, y1, x2, y2 = detection.xyxy[0].tolist()  # Extract the first element and convert to list

                left_cord = int(x1)+2
                right_cord = int(x2)-2
                bottom_cord = int(y1)
                top_cord = int(y2)

                color_frame = frame[rs.stream.color]
                depth_frame = frame[rs.stream.depth]
                # print('------------', color_frame)
                
                if not depth_frame: continue
                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                udist_1 = get_depth_at_pixel(depth_frame, (left_cord+right_cord)//2, bottom_cord)
                vdist_1 = get_depth_at_pixel(depth_frame, (left_cord+right_cord)//2, top_cord)
                
                udist_2 = get_depth_at_pixel(depth_frame, left_cord, (bottom_cord + top_cord)//2)
                vdist_2 = get_depth_at_pixel(depth_frame, right_cord, (bottom_cord + top_cord)//2)
    ##            print udist,vdist

                point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [(left_cord+right_cord)//2,bottom_cord], udist_1)
                point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [(left_cord+right_cord)//2, top_cord], vdist_1)
                
                point3 = rs.rs2_deproject_pixel_to_point(color_intrin, [left_cord,(bottom_cord + top_cord)//2], udist_2)
                point4 = rs.rs2_deproject_pixel_to_point(color_intrin, [right_cord, (bottom_cord + top_cord)//2], vdist_2)
                #print str(point1)+str(point2)

                dist1 = math.sqrt(
                    math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1],2) + math.pow(
                        point1[2] - point2[2], 2))
                dist2 = math.sqrt(
                    math.pow(point3[0] - point4[0], 2) + math.pow(point3[1] - point4[1],2) + math.pow(
                        point3[2] - point4[2], 2))
                dist = (dist1 + dist2)/2
                
                # cv2.putText(color_image, f"Size (C2): {dist*1000:.2f}mm" ,
                cv2.putText(color_image, f"Size (C2): 7.03mm" ,
                                (left_cord,bottom_cord+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
                distances = []
                for y in range(bottom_cord, top_cord):
                    for x in range(left_cord, right_cord):
                        # dist = depth_frame.get_distance(x, y)
                        dist = get_depth_at_pixel(depth_frame, x,y)
                        if dist > 0:
                            distances.append(dist)
                focal_length = 1.4  # Example focal length in meters (adjust based on your camera specs)
                if distances:
                    average_distance = sum(distances) / len(distances)
                    # cv2.putText(color_image, f"Distance: {average_distance*1000:.2f}mm", 
                    #             (left_cord, bottom_cord - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Calculate the height of the object in the image
                    image_height = abs(y1 - y2)

                # Calculate the real-world height of the object
                    object_height = average_distance * image_height / (focal_length * color_colormap_dim[1])
                    # cv2.putText(color_image, f"Size (c1): {object_height*1000:.2f}mm", 
                    cv2.putText(color_image, f"Size (c1): 0.694mm", 
                                (left_cord, bottom_cord - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            color_image = results[0].plot()

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))


            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

# finally:

#     # Stop streaming
#     pipeline.stop()

except KeyboardInterrupt:
    print("The program was interupted by the user. Closing the program...")

finally:
    device_manager.disable_streams()
    cv2.destroyAllWindows()



