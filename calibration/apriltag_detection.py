# Need to continuously output the apriltag detection results (pose)

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
import os
import sys
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

def initialize_pipeline(serial=None):
    # Load camera serial from config
    config_path = os.path.join(os.path.dirname(__file__), '../config/device_port.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if serial is None:
        serial = config.get('camera_serial')
    print(f"Using camera serial: {serial}")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    return pipeline, profile, align

def get_camera_intrinsics(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return intr.fx, intr.fy, intr.ppx, intr.ppy, intr.coeffs

def process_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image

def draw_green_box(color_image, tag):
    """Draw a green box around the AprilTag without displaying the ID."""
    for idx in range(len(tag.corners)):
        cv2.line(color_image, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 2)

def rotation_matrix_to_euler_angles(rotation_matrix):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees using scipy."""
    r = R.from_matrix(rotation_matrix)
    # Get Euler angles in XYZ convention (roll, pitch, yaw)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    return roll, pitch, yaw

def main():
    pipeline, profile, align = initialize_pipeline()
    fx, fy, cx, cy, _ = get_camera_intrinsics(profile)
    detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
    tag_size = 0.0792  # Set the tag size in meters

    print("Displaying camera stream with AprilTag detection. Press 'q' to quit.")

    while True:
        color_image, depth_image = process_frames(pipeline, align)
        if color_image is None or depth_image is None:
            continue

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray_image, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy], tag_size=tag_size)

        if tags:
            for tag in tags:
                draw_green_box(color_image, tag)
                print("=" * 60)
                print(f"Tag ID: {tag.tag_id}")
                print(f"Center (px): ({tag.center[0]:.2f}, {tag.center[1]:.2f})")
                print(f"Translation (m):")
                print(f"  X: {tag.pose_t[0][0]:8.4f}")
                print(f"  Y: {tag.pose_t[1][0]:8.4f}")
                print(f"  Z: {tag.pose_t[2][0]:8.4f}")
                print(f"Rotation Matrix:")
                print(f"  [{tag.pose_R[0][0]:7.4f}, {tag.pose_R[0][1]:7.4f}, {tag.pose_R[0][2]:7.4f}]")
                print(f"  [{tag.pose_R[1][0]:7.4f}, {tag.pose_R[1][1]:7.4f}, {tag.pose_R[1][2]:7.4f}]")
                print(f"  [{tag.pose_R[2][0]:7.4f}, {tag.pose_R[2][1]:7.4f}, {tag.pose_R[2][2]:7.4f}]")
                
                # Convert rotation matrix to Euler angles
                roll, pitch, yaw = rotation_matrix_to_euler_angles(tag.pose_R)
                print(f"Euler Angles (degrees):")
                print(f"  Roll:  {roll:7.2f}°")
                print(f"  Pitch: {pitch:7.2f}°")
                print(f"  Yaw:   {yaw:7.2f}°")
                
                # According to the AprilTag documentation (https://pupil-apriltags.readthedocs.io/en/latest/api.html),
                # tag.pose_R is the rotation matrix and tag.pose_t is the translation vector.

        cv2.imshow("AprilTag Detection", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
