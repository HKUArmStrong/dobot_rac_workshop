#!/usr/bin/env python3
"""
RealSense Utility Functions
============================
Common functions for working with Intel RealSense cameras.
Simplifies pipeline setup, frame processing, and coordinate conversions.
"""

import pyrealsense2 as rs
import numpy as np
import cv2


def initialize_pipeline(width=640, height=480, fps=30, serial_number=None):
    """
    Initialize RealSense pipeline with color and depth streams.
    
    Args:
        width: Frame width (default: 640)
        height: Frame height (default: 480)
        fps: Frames per second (default: 30)
        serial_number: Specific device serial (optional)
        
    Returns:
        (pipeline, profile, config): Initialized pipeline, profile, and config objects
    """
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable specific device if serial provided
    if serial_number:
        config.enable_device(serial_number)
    
    # Enable streams
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    
    # Start streaming
    profile = pipeline.start(config)
    
    return pipeline, profile, config


def get_camera_intrinsics(profile, stream_type=rs.stream.depth):
    """
    Get camera intrinsic parameters from profile.
    
    Args:
        profile: RealSense pipeline profile
        stream_type: Stream to get intrinsics from (default: depth)
        
    Returns:
        intrinsics: Camera intrinsics object
    """
    stream_profile = profile.get_stream(stream_type)
    intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
    return intrinsics


def get_aligned_frames(pipeline, align):
    """
    Get and align color and depth frames.
    
    Args:
        pipeline: RealSense pipeline
        align: Alignment object
        
    Returns:
        (color_frame, depth_frame): Aligned frames, or (None, None) if invalid
    """
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        return None, None
    
    return color_frame, depth_frame


def frames_to_numpy(color_frame, depth_frame):
    """
    Convert RealSense frames to numpy arrays.
    
    Args:
        color_frame: RealSense color frame
        depth_frame: RealSense depth frame
        
    Returns:
        (color_image, depth_image): Numpy arrays
    """
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image


def depth_to_colormap(depth_image, alpha=0.03, colormap=cv2.COLORMAP_JET):
    """
    Convert depth image to colormap for visualization.
    
    Args:
        depth_image: Depth image as numpy array
        alpha: Scale factor for depth (default: 0.03)
        colormap: OpenCV colormap to use (default: COLORMAP_JET)
        
    Returns:
        depth_colormap: Colorized depth image
    """
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=alpha),
        colormap
    )
    return depth_colormap


def pixel_to_3d(intrinsics, pixel_x, pixel_y, depth_frame):
    """
    Convert 2D pixel coordinate to 3D world coordinate.
    
    Args:
        intrinsics: Camera intrinsics object
        pixel_x: X coordinate in pixels
        pixel_y: Y coordinate in pixels
        depth_frame: RealSense depth frame
        
    Returns:
        [X, Y, Z] in meters, or None if invalid depth
    """
    depth = depth_frame.get_distance(pixel_x, pixel_y)
    
    if depth == 0:
        return None
    
    point_3d = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [pixel_x, pixel_y],
        depth
    )
    
    return point_3d


def point_3d_to_pixel(intrinsics, point_3d):
    """
    Convert 3D world coordinate to 2D pixel coordinate.
    
    Args:
        intrinsics: Camera intrinsics object
        point_3d: [X, Y, Z] in meters
        
    Returns:
        [pixel_x, pixel_y]: Pixel coordinates
    """
    pixel = rs.rs2_project_point_to_pixel(intrinsics, point_3d)
    return [int(pixel[0]), int(pixel[1])]


def get_depth_scale(profile):
    """
    Get depth scale factor (converts depth units to meters).
    
    Args:
        profile: RealSense pipeline profile
        
    Returns:
        depth_scale: Scale factor
    """
    depth_sensor = profile.get_device().first_depth_sensor()
    return depth_sensor.get_depth_scale()


def print_camera_info(intrinsics):
    """
    Print camera intrinsic parameters.
    
    Args:
        intrinsics: Camera intrinsics object
    """
    print("Camera Intrinsics:")
    print(f"  Resolution: {intrinsics.width} x {intrinsics.height}")
    print(f"  Principal Point (cx, cy): ({intrinsics.ppx:.2f}, {intrinsics.ppy:.2f})")
    print(f"  Focal Length (fx, fy): ({intrinsics.fx:.2f}, {intrinsics.fy:.2f})")
    print(f"  Distortion Model: {intrinsics.model}")


def list_connected_devices():
    """
    List all connected RealSense devices.
    
    Returns:
        devices: List of device serial numbers
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    device_list = []
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"Device {i}: {name} (Serial: {serial})")
        device_list.append(serial)
    
    return device_list
