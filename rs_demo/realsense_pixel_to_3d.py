#!/usr/bin/env python3
"""
Demo 2: Pixel to 3D Coordinate Conversion
==========================================
This demo shows how to:
1. Convert pixel coordinates (x, y) to 3D world coordinates (X, Y, Z)
2. Use camera intrinsics and depth information
3. Handle mouse clicks to interactively explore 3D coordinates

Click on the image to see the 3D coordinates of that pixel.
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
    pixel_to_3d,
    print_camera_info
)


class RealSense3DConverter:
    """Interactive demo for converting 2D pixels to 3D coordinates"""
    
    def __init__(self):
        # Initialize pipeline
        self.pipeline, profile, config = initialize_pipeline(width=640, height=480, fps=30)
        
        # Get camera intrinsics
        self.intrinsics = get_camera_intrinsics(profile)
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)
        
        # Store clicked point
        self.clicked_point = None
        
        print("=" * 70)
        print("RealSense SDK Demo 2: Pixel to 3D Conversion")
        print("=" * 70)
        print()
        print_camera_info(self.intrinsics)
        print("\n" + "=" * 70)
        print("HOW PIXEL TO 3D CONVERSION WORKS:")
        print("=" * 70)
        print("""
                The conversion from 2D pixel (x, y) to 3D point (X, Y, Z) uses:

                1. Depth value at pixel (x, y): Z = depth(x, y) in meters
                2. Camera intrinsics: focal length (fx, fy) and principal point (cx, cy)
                3. Pinhole camera model equations:

                X = (x - cx) * Z / fx
                Y = (y - cy) * Z / fy
                Z = depth(x, y)

                Where:
                - (x, y) = pixel coordinates
                - (cx, cy) = camera principal point (image center)
                - (fx, fy) = camera focal lengths in pixels
                - Z = depth value at that pixel
                - (X, Y, Z) = 3D coordinates in camera frame (meters)

                The RealSense SDK provides a helper function rs.rs2_deproject_pixel_to_point()
                that performs this calculation automatically.
                """)
        print("=" * 70)
        print("\nClick on the image to see 3D coordinates!")
        print("Press 'q' to quit\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select pixels"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
    
    def draw_crosshair(self, image, x, y, size=20, color=(0, 255, 0), thickness=2):
        """Draw a crosshair at the specified pixel location"""
        cv2.line(image, (x - size, y), (x + size, y), color, thickness)
        cv2.line(image, (x, y - size), (x, y + size), color, thickness)
        cv2.circle(image, (x, y), 5, color, -1)
    
    def run(self):
        """Main loop"""
        # Create window and set mouse callback
        window_name = 'RealSense: Click to get 3D coordinates'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        try:
            while True:
                # Get aligned frames
                color_frame, depth_frame = get_aligned_frames(self.pipeline, self.align)
                
                if color_frame is None or depth_frame is None:
                    continue
                
                # Convert to numpy array
                color_image, _ = frames_to_numpy(color_frame, depth_frame)
                
                # If a point was clicked, process it
                if self.clicked_point is not None:
                    x, y = self.clicked_point
                    
                    # Convert pixel to 3D using utility function
                    point_3d = pixel_to_3d(self.intrinsics, x, y, depth_frame)
                    
                    # Draw crosshair at clicked location
                    self.draw_crosshair(color_image, x, y)
                    
                    # Display information
                    if point_3d is not None:
                        X, Y, Z = point_3d
                        
                        # Convert to millimeters for easier reading
                        X_mm, Y_mm, Z_mm = X * 1000, Y * 1000, Z * 1000
                        
                        # Print to console
                        print(f"\n{'='*60}")
                        print(f"Pixel: ({x}, {y})")
                        print(f"3D Point in Camera Frame:")
                        print(f"  X = {X_mm:7.1f} mm")
                        print(f"  Y = {Y_mm:7.1f} mm")
                        print(f"  Z = {Z_mm:7.1f} mm (depth)")
                        print(f"  Distance: {np.sqrt(X**2 + Y**2 + Z**2)*1000:.1f} mm")
                        print(f"{'='*60}")
                        
                        # Draw info on image
                        info_text = [
                            f"Pixel: ({x}, {y})",
                            f"3D: X={X_mm:.1f}mm Y={Y_mm:.1f}mm Z={Z_mm:.1f}mm",
                            f"Distance: {np.sqrt(X**2 + Y**2 + Z**2)*1000:.1f}mm"
                        ]
                        
                        # Background for text
                        cv2.rectangle(color_image, (x + 15, y - 45), (x + 450, y + 15), 
                                    (0, 0, 0), -1)
                        cv2.rectangle(color_image, (x + 15, y - 45), (x + 450, y + 15), 
                                    (0, 255, 0), 2)
                        
                        for i, text in enumerate(info_text):
                            cv2.putText(color_image, text, (x + 20, y - 25 + i * 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        # Invalid depth
                        cv2.putText(color_image, "Invalid depth at this pixel", 
                                  (x + 20, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print(f"\nPixel ({x}, {y}): Invalid depth (too close or no data)")
                
                # Instructions
                cv2.putText(color_image, "Click anywhere to get 3D coordinates", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(color_image, "Press 'q' to quit", 
                          (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw center crosshair for reference
                center_x, center_y = 320, 240
                cv2.line(color_image, (center_x - 10, center_y), (center_x + 10, center_y), 
                        (128, 128, 128), 1)
                cv2.line(color_image, (center_x, center_y - 10), (center_x, center_y + 10), 
                        (128, 128, 128), 1)
                cv2.putText(color_image, "Center", (center_x + 15, center_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                
                # Show image
                cv2.imshow(window_name, color_image)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Clear clicked point
                    self.clicked_point = None
                    print("\nCleared selection")
        
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("\n" + "=" * 70)
            print("Demo completed!")
            print("=" * 70)


def main():
    converter = RealSense3DConverter()
    converter.run()


if __name__ == "__main__":
    main()
