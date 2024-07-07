import cv2
import math
from ultralytics import YOLOv10

# Load the YOLOv8 model
model = YOLOv10("weights/yolov10_best.pt")

# Depth Cameras D435
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
# calib camera
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
    
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
# preprocess - color 

        # Run YOLOv8 inference on the frame
        results = model(color_image, conf=0.1)
        
        # Extract the labels and positions
        detections = results[0].boxes
        
    
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

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # print('----------', color_frame)
            if not depth_frame: continue
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

            udist = depth_frame.get_distance((left_cord+right_cord)//2, bottom_cord)
            vdist = depth_frame.get_distance((left_cord+right_cord)//2, top_cord)
            
            udist_1 = depth_frame.get_distance(left_cord, (top_cord+bottom_cord)//2)
            vdist_1 = depth_frame.get_distance(right_cord, (top_cord+bottom_cord)//2)
##            print udist,vdist

            point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [(left_cord+right_cord)//2,bottom_cord], udist)
            point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [(left_cord+right_cord)//2, top_cord], vdist)
            #print str(point1)+str(point2)
            point3 = rs.rs2_deproject_pixel_to_point(color_intrin, [left_cord, (top_cord+bottom_cord)//2], udist)
            point4 = rs.rs2_deproject_pixel_to_point(color_intrin, [right_cord, (top_cord+bottom_cord)//2], vdist)
            
            dist_1 = math.sqrt(
                math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1],2) + math.pow(
                    point1[2] - point2[2], 2))
            dist_2 = math.sqrt(
                math.pow(point3[0] - point4[0], 2) + math.pow(point3[1] - point4[1],2) + math.pow(
                    point3[2] - point4[2], 2))
            dist = (dist_1 + dist_2) /2
            
            cv2.putText(color_image, f"Size_2: {dist*1000:.2f}mm" ,
            # cv2.putText(color_image, f"Size_2: 3.45mm" ,
                            (left_cord,bottom_cord+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
            distances = []
            for y in range(bottom_cord, top_cord):
                for x in range(left_cord, right_cord):
                    dist = depth_frame.get_distance(x, y)
                    if dist > 0:
                        distances.append(dist)
            focal_length = 1.0  # Example focal length in meters (adjust based on your camera specs)
            if distances:
                average_distance = sum(distances) / len(distances)
                # cv2.putText(color_image, f"Distance: {average_distance*1000:.2f}mm", 
                #             (left_cord, bottom_cord - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate the height of the object in the image
                image_height = abs(y1 - y2)


               # Calculate the real-world height of the object
                object_height = average_distance * image_height / (focal_length * color_colormap_dim[1])
                cv2.putText(color_image, f"Size_1: {object_height*1000:.2f}mm", 
                # cv2.putText(color_image, f"Size_1: 4.03mm", 
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

finally:

    # Stop streaming
    pipeline.stop()


