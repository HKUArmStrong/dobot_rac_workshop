#!/usr/bin/env python3
"""
Demo 1: Basic RealSense Camera Setup
=====================================
This demo shows how to:
1. Initialize the RealSense pipeline
2. Configure color and depth streams
3. Capture and display frames
4. Properly cleanup resources

Press 'q' to quit.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from realsense_utils import (
    initialize_pipeline,
    get_camera_intrinsics,
    get_aligned_frames,
    frames_to_numpy,
    depth_to_colormap,
    print_camera_info
)


def main():
    print("=" * 70)
    print("RealSense SDK Demo 1: Basic Camera Setup")
    print("=" * 70)
    
    # ========================================
    # Step 1-3: Initialize pipeline with color and depth streams
    # ========================================
    print("\n[Steps 1-3] Initializing pipeline...")
    pipeline, profile, config = initialize_pipeline(width=640, height=480, fps=30)
    print("  - Color: 640x480 @ 30fps (BGR8)")
    print("  - Depth: 640x480 @ 30fps (Z16)")
    print("  - Pipeline started successfully!\n")
    
    # ========================================
    # Step 4: Get camera intrinsics
    # ========================================
    print("[Step 4] Camera Intrinsics:")
    intrinsics = get_camera_intrinsics(profile)
    print_camera_info(intrinsics)
    
    # ========================================
    # Step 5: Create alignment object
    # ========================================
    print("\n[Step 5] Creating alignment object (depth → color)")
    align = rs.align(rs.stream.color)
    
    print("\nStarting live view... Press 'q' to quit\n")
    
    try:
        frame_count = 0
        while True:
            # ========================================
            # Step 6-7: Get and align frames
            # ========================================
            color_frame, depth_frame = get_aligned_frames(pipeline, align)
            
            # Validate frames
            if color_frame is None or depth_frame is None:
                continue
            
            # ========================================
            # Step 8: Convert frames to numpy arrays
            # ========================================
            color_image, depth_image = frames_to_numpy(color_frame, depth_frame)
            
            # ========================================
            # Step 9: Create visualization
            # ========================================
            # Apply colormap to depth image
            depth_colormap = depth_to_colormap(depth_image)
            
            # Stack images horizontally for display
            images = np.hstack((color_image, depth_colormap))
            
            # Add text overlay with frame info
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Frame {frame_count} captured")
            
            # Display instructions
            cv2.putText(images, "Color Image | Depth Colormap", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(images, "Press 'q' to quit", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show images
            cv2.imshow('RealSense Demo 1: Basic Setup', images)
            
            # ========================================
            # Step 10: Handle user input
            # ========================================
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
    
    finally:
        # ========================================
        # Step 11: Cleanup
        # ========================================
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped and resources released.")
        print("=" * 70)


if __name__ == "__main__":
    main()
