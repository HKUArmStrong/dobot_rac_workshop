#!/usr/bin/env python3
"""
Real-time 3D visualization combining robot arm, AprilTag, and camera frames.
All coordinate frames are expressed relative to the robot base frame.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
from pydobotplus import Dobot
import sys


def get_dobot_port():
    """Load Dobot port from config file"""
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]


def initialize_pipeline(serial=None):
    """Initialize RealSense pipeline"""
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
    """Get camera intrinsic parameters"""
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return intr.fx, intr.fy, intr.ppx, intr.ppy, intr.coeffs


def process_frames(pipeline, align):
    """Process camera frames"""
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image


def get_robot_arm_matrix(pose):
    """
    Build a 4x4 transformation matrix (will transform point under gripper frame to base frame) in mm from a robot pose object.
    Assumes planar robot (rotation about Z only).
    """
    # YOUR CODE STARTS HERE
    raise NotImplementedError("Implement get_robot_arm_matrix function")
    # YOUR CODE ENDS HERE


def get_tag_to_camera_matrix(tag):
    """
    Build a 4x4 transformation matrix (will transform points under tag frame to camera frame) in mm from a detected tag object.
    """
    # YOUR CODE STARTS HERE
    raise NotImplementedError("Implement get_tag_to_camera_matrix function")
    # YOUR CODE ENDS HERE


def get_tag_to_gripper_matrix():
    """
    Build a 4x4 transformation matrix (will transform points under tag frame to gripper frame) in mm.
    Tag is 153mm below gripper, 30mm forward along gripper X.
    X-axes aligned, but Z-axis flipped.
    """
    # YOUR CODE STARTS HERE
    raise NotImplementedError("Implement get_tag_to_gripper_matrix function")
    # YOUR CODE ENDS HERE


def draw_coordinate_frame(ax, T, scale=50.0, label="", colors=None):
    """
    Draw a coordinate frame at the position/orientation defined by transformation matrix T.
    
    Args:
        ax: matplotlib 3D axis
        T: 4x4 transformation matrix
        scale: length of axes arrows (mm)
        label: label prefix for the frame (e.g., "Base", "EE", "Tag")
        colors: list of colors for [x, y, z] axes, default is ['r', 'g', 'b']
    """
    if colors is None:
        colors = ['red', 'green', 'blue']
    
    # Extract position
    position = T[:3, 3]
    
    # Define unit axes in local frame
    axes_local = np.array([
        [scale, 0, 0, 1],  # X-axis
        [0, scale, 0, 1],  # Y-axis
        [0, 0, scale, 1]   # Z-axis
    ])
    
    # Transform axes to world frame
    axes_world = (T @ axes_local.T).T
    
    # Draw each axis
    axis_names = ['X', 'Y', 'Z']
    for i, (color, name) in enumerate(zip(colors, axis_names)):
        # Vector from position to axis endpoint
        vector = axes_world[i, :3] - position
        
        # Draw arrow
        ax.quiver(position[0], position[1], position[2],
                 vector[0], vector[1], vector[2],
                 color=color, arrow_length_ratio=0.15, linewidth=2.5, alpha=0.9)
        
        # Add text label at the end of arrow
        text_pos = axes_world[i, :3]
        ax.text(text_pos[0], text_pos[1], text_pos[2], 
               f"{label}_{name}", fontsize=9, color=color, fontweight='bold')


class SetupVisualizer:
    """Real-time 3D visualization of complete setup: robot, camera, and AprilTag"""
    
    def __init__(self, device, pipeline, align, detector, fx, fy, cx, cy, tag_size):
        self.device = device
        self.pipeline = pipeline
        self.align = align
        self.detector = detector
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.tag_size = tag_size
        
        # Known gripper to tag transformation (in mm)
        self.gripper_T_tag = get_tag_to_gripper_matrix()
        
        # Setup figure with two subplots: 3D view and camera view
        self.fig = plt.figure(figsize=(16, 8))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_camera = self.fig.add_subplot(122)
        
        # Set initial viewing angle
        self.ax_3d.view_init(elev=20.0, azim=-60.0)
        
        # Text box for displaying transformation matrices
        self.info_text = self.fig.text(0.02, 0.98, '', fontsize=8, family='monospace',
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Store latest tag detection
        self.latest_tag = None
        
    def update(self, frame):
        """Update function called by animation"""
        try:
            # ========================================
            # 1. Get camera frames and detect AprilTags
            # ========================================
            color_image, depth_image = process_frames(self.pipeline, self.align)
            if color_image is None or depth_image is None:
                return self.ax_3d, self.ax_camera
            
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray_image, estimate_tag_pose=True, 
                                       camera_params=[self.fx, self.fy, self.cx, self.cy], 
                                       tag_size=self.tag_size)
            
            # Draw detection on camera image
            display_image = color_image.copy()
            if tags:
                self.latest_tag = tags[0]  # Use first detected tag
                tag = self.latest_tag
                
                # Draw green box around tag
                for idx in range(len(tag.corners)):
                    cv2.line(display_image, 
                            tuple(tag.corners[idx - 1, :].astype(int)), 
                            tuple(tag.corners[idx, :].astype(int)), 
                            (0, 255, 0), 2)
                cv2.circle(display_image, tuple(tag.center.astype(int)), 5, (0, 0, 255), -1)
                cv2.putText(display_image, f"ID: {tag.tag_id}", 
                           (int(tag.center[0]), int(tag.center[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # ========================================
            # 2. Update camera view
            # ========================================
            self.ax_camera.clear()
            self.ax_camera.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            self.ax_camera.set_title('Camera View with AprilTag Detection', fontsize=12, fontweight='bold')
            self.ax_camera.axis('off')
            
            # ========================================
            # 3. Get robot pose
            # ========================================
            pose = self.device.get_pose()
            base_T_gripper = get_robot_arm_matrix(pose)
            
            # ========================================
            # 4. Update 3D visualization
            # ========================================
            # Save current view angle before clearing
            elev = self.ax_3d.elev
            azim = self.ax_3d.azim
            
            self.ax_3d.clear()
            
            # ========================================
            # Draw BASE FRAME at origin (fixed reference)
            # ========================================
            T_base = np.eye(4)
            draw_coordinate_frame(self.ax_3d, T_base, scale=80.0, label="Base",
                                colors=['red', 'green', 'blue'])
            
            # Base origin marker
            self.ax_3d.scatter([0], [0], [0], c='black', s=150, marker='s', 
                          edgecolors='black', linewidths=2, label='Robot Base')
            
            # ========================================
            # Draw END-EFFECTOR/GRIPPER FRAME
            # ========================================
            draw_coordinate_frame(self.ax_3d, base_T_gripper, scale=50.0, label="Gripper",
                                colors=['darkred', 'darkgreen', 'darkblue'])
            
            # Gripper position marker
            gripper_pos = base_T_gripper[:3, 3]
            self.ax_3d.scatter([gripper_pos[0]], [gripper_pos[1]], [gripper_pos[2]], 
                          c='orange', s=120, marker='o', edgecolors='black', 
                          linewidths=2, label='Gripper')
            
            # Connection line from base to gripper
            self.ax_3d.plot([0, gripper_pos[0]], 
                        [0, gripper_pos[1]], 
                        [0, gripper_pos[2]], 
                        'gray', linestyle='--', linewidth=1.5, alpha=0.6)
            
            # ========================================
            # Draw APRILTAG FRAME (fixed relative to gripper)
            # ========================================
            base_T_tag = base_T_gripper @ self.gripper_T_tag
            draw_coordinate_frame(self.ax_3d, base_T_tag, scale=50.0, label="Tag",
                                colors=['crimson', 'limegreen', 'dodgerblue'])
            
            # Tag position marker
            tag_pos = base_T_tag[:3, 3]
            self.ax_3d.scatter([tag_pos[0]], [tag_pos[1]], [tag_pos[2]], 
                          c='purple', s=120, marker='^', edgecolors='black', 
                          linewidths=2, label='AprilTag')
            
            # Connection line from gripper to tag
            self.ax_3d.plot([gripper_pos[0], tag_pos[0]], 
                        [gripper_pos[1], tag_pos[1]], 
                        [gripper_pos[2], tag_pos[2]], 
                        'purple', linestyle=':', linewidth=1.5, alpha=0.6)
            
            # ========================================
            # Draw CAMERA FRAME (if tag detected)
            # Compute camera position from: base_T_camera = base_T_tag @ inv(cam_T_tag)
            # ========================================
            if self.latest_tag is not None:
                cam_T_tag = get_tag_to_camera_matrix(self.latest_tag)
                tag_T_camera = np.linalg.inv(cam_T_tag)
                base_T_camera = base_T_tag @ tag_T_camera
                
                draw_coordinate_frame(self.ax_3d, base_T_camera, scale=60.0, label="Cam",
                                    colors=['salmon', 'lightgreen', 'skyblue'])
                
                # Camera position marker
                camera_pos = base_T_camera[:3, 3]
                self.ax_3d.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], 
                              c='cyan', s=120, marker='D', edgecolors='black', 
                              linewidths=2, label='Camera')
                
                # Connection line from base to camera
                self.ax_3d.plot([0, camera_pos[0]], 
                            [0, camera_pos[1]], 
                            [0, camera_pos[2]], 
                            'cyan', linestyle='-.', linewidth=1.5, alpha=0.5)
                
                # Connection line from camera to tag
                self.ax_3d.plot([camera_pos[0], tag_pos[0]], 
                            [camera_pos[1], tag_pos[1]], 
                            [camera_pos[2], tag_pos[2]], 
                            'yellow', linestyle=':', linewidth=2, alpha=0.7)
            
            # ========================================
            # Configure plot appearance
            # ========================================
            self.ax_3d.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
            self.ax_3d.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
            self.ax_3d.set_zlabel('Z (mm)', fontsize=11, fontweight='bold')
            self.ax_3d.set_title('Robot Setup Visualization\n(Base Frame Reference)', 
                            fontsize=12, fontweight='bold')
            
            # Set fixed axis limits for workspace
            self.ax_3d.set_xlim([-350, 350])
            self.ax_3d.set_ylim([-350, 350])
            self.ax_3d.set_zlim([0, 400])
            
            # Equal aspect ratio
            self.ax_3d.set_box_aspect([1, 1, 1])
            
            # Restore user's view angle
            self.ax_3d.view_init(elev=elev, azim=azim)
            
            # Legend
            self.ax_3d.legend(loc='upper right', fontsize=9)
            
            # ========================================
            # Display transformation matrix information
            # ========================================
            info = f"""Robot Pose: X={pose.position.x:.1f}, Y={pose.position.y:.1f}, Z={pose.position.z:.1f} mm, R={pose.position.r:.1f}°
                base_T_gripper:
                [{base_T_gripper[0,0]:7.4f} {base_T_gripper[0,1]:7.4f} {base_T_gripper[0,2]:7.4f} | {base_T_gripper[0,3]:7.1f}]
                [{base_T_gripper[1,0]:7.4f} {base_T_gripper[1,1]:7.4f} {base_T_gripper[1,2]:7.4f} | {base_T_gripper[1,3]:7.1f}]
                [{base_T_gripper[2,0]:7.4f} {base_T_gripper[2,1]:7.4f} {base_T_gripper[2,2]:7.4f} | {base_T_gripper[2,3]:7.1f}]
                [{base_T_gripper[3,0]:7.4f} {base_T_gripper[3,1]:7.4f} {base_T_gripper[3,2]:7.4f} | {base_T_gripper[3,3]:7.1f}]

                base_T_tag (gripper_to_tag offset: [30, 0, 153] mm):
                [{base_T_tag[0,0]:7.4f} {base_T_tag[0,1]:7.4f} {base_T_tag[0,2]:7.4f} | {base_T_tag[0,3]:7.1f}]
                [{base_T_tag[1,0]:7.4f} {base_T_tag[1,1]:7.4f} {base_T_tag[1,2]:7.4f} | {base_T_tag[1,3]:7.1f}]
                [{base_T_tag[2,0]:7.4f} {base_T_tag[2,1]:7.4f} {base_T_tag[2,2]:7.4f} | {base_T_tag[2,3]:7.1f}]
                [{base_T_tag[3,0]:7.4f} {base_T_tag[3,1]:7.4f} {base_T_tag[3,2]:7.4f} | {base_T_tag[3,3]:7.1f}]
                """
            
            if self.latest_tag is not None:
                tag = self.latest_tag
                cam_T_tag = get_tag_to_camera_matrix(tag)
                r = R.from_matrix(tag.pose_R)
                roll, pitch, yaw = r.as_euler('xyz', degrees=True)
                
                # Compute camera position in base frame
                tag_T_camera = np.linalg.inv(cam_T_tag)
                base_T_camera = base_T_tag @ tag_T_camera
                
                info += f"""
                        AprilTag ID: {tag.tag_id} (detected in camera frame)
                        Center (px): ({tag.center[0]:.1f}, {tag.center[1]:.1f})
                        Distance from camera: {np.linalg.norm(cam_T_tag[:3, 3]):.1f} mm

                        cam_T_tag:
                        [{cam_T_tag[0,0]:7.4f} {cam_T_tag[0,1]:7.4f} {cam_T_tag[0,2]:7.4f} | {cam_T_tag[0,3]:7.1f}]
                        [{cam_T_tag[1,0]:7.4f} {cam_T_tag[1,1]:7.4f} {cam_T_tag[1,2]:7.4f} | {cam_T_tag[1,3]:7.1f}]
                        [{cam_T_tag[2,0]:7.4f} {cam_T_tag[2,1]:7.4f} {cam_T_tag[2,2]:7.4f} | {cam_T_tag[2,3]:7.1f}]
                        [{cam_T_tag[3,0]:7.4f} {cam_T_tag[3,1]:7.4f} {cam_T_tag[3,2]:7.4f} | {cam_T_tag[3,3]:7.1f}]
                        Euler: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°

                        base_T_camera (computed from detection):
                        [{base_T_camera[0,0]:7.4f} {base_T_camera[0,1]:7.4f} {base_T_camera[0,2]:7.4f} | {base_T_camera[0,3]:7.1f}]
                        [{base_T_camera[1,0]:7.4f} {base_T_camera[1,1]:7.4f} {base_T_camera[1,2]:7.4f} | {base_T_camera[1,3]:7.1f}]
                        [{base_T_camera[2,0]:7.4f} {base_T_camera[2,1]:7.4f} {base_T_camera[2,2]:7.4f} | {base_T_camera[2,3]:7.1f}]
                        [{base_T_camera[3,0]:7.4f} {base_T_camera[3,1]:7.4f} {base_T_camera[3,2]:7.4f} | {base_T_camera[3,3]:7.1f}]
                        """
            else:
                info += "\nAprilTag: Not detected (camera pose cannot be computed)"
            
            self.info_text.set_text(info)
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()
        
        return self.ax_3d, self.ax_camera
    
    def run(self):
        """Start the animation"""
        ani = FuncAnimation(self.fig, self.update, interval=100, blit=False, cache_frame_data=False)
        plt.show()


def main():
    """Main entry point"""
    try:
        # Connect to robot
        port = get_dobot_port()
        print(f"Connecting to Dobot on port: {port}")
        device = Dobot(port=port)
        print("Connected successfully!")
        device.home()
        
        # Initialize camera
        print("Initializing RealSense camera...")
        pipeline, profile, align = initialize_pipeline()
        fx, fy, cx, cy, _ = get_camera_intrinsics(profile)
        print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        # Initialize AprilTag detector
        print("Initializing AprilTag detector...")
        detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0, 
                           quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
        tag_size = 0.0792  # Set the tag size in meters
        print(f"Tag size: {tag_size} m = {tag_size * 1000} mm")
        
        print("\nStarting visualization...")
        print("="*70)
        print("Visualizing robot setup with coordinate frames:")
        print("  - Base frame (robot base) at origin")
        print("  - Gripper/end-effector frame")
        print("  - AprilTag frame (fixed 153mm below gripper)")
        print("  - Camera frame (computed from AprilTag detection)")
        print("Close the window to exit.")
        print("="*70)
        
        # Start visualization
        visualizer = SetupVisualizer(device, pipeline, align, detector, fx, fy, cx, cy, tag_size)
        visualizer.run()
        
        # Cleanup
        pipeline.stop()
        device.close()
        print("\nCamera stopped. Devices closed. Visualization closed.")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
