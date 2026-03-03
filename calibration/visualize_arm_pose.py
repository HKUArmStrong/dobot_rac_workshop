#!/usr/bin/env python3
"""
Real-time visualization of Dobot arm using transformation matrices.
Displays base and end-effector coordinate frames using matplotlib.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pydobotplus import Dobot
import sys


def get_dobot_port():
    """Load Dobot port from config file"""
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]


def get_robot_arm_matrix(pose):
    """
    Build a 4x4 transformation matrix (will transform points under gripper frame to base frame) in mm from a robot pose object.
    Assumes planar robot (rotation about Z only).
    """
    # YOUR CODE STARTS HERE
    raise NotImplementedError("Implement get_robot_arm_matrix function")
    # YOUR CODE ENDS HERE


def draw_coordinate_frame(ax, T, scale=50.0, label="", colors=None):
    """
    Draw a coordinate frame at the position/orientation defined by transformation matrix T.
    
    Args:
        ax: matplotlib 3D axis
        T: 4x4 transformation matrix
        scale: length of axes arrows
        label: label prefix for the frame (e.g., "Base", "EE")
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


class ArmVisualizer:
    """Real-time 3D visualization of robot arm coordinate frames"""
    
    def __init__(self, device):
        self.device = device
        
        # Setup figure and 3D axis
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Text box for displaying transformation matrix
        self.info_text = self.fig.text(0.02, 0.98, '', fontsize=9, family='monospace',
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def update(self, frame):
        """Update function called by animation"""
        try:
            # Clear previous frame
            self.ax.clear()
            
            # Get current pose
            pose = self.device.get_pose()
            x, y, z, r = pose.position.x, pose.position.y, pose.position.z, pose.position.r
            
            # ========================================
            # Calculate transformation matrix (Base → End-Effector)
            # ========================================
            base_T_gripper = get_robot_arm_matrix(pose)
            
            # ========================================
            # 1. Draw BASE FRAME at origin
            # ========================================
            T_base = np.eye(4)  # Identity matrix = origin
            draw_coordinate_frame(self.ax, T_base, scale=50.0, label="Base",
                                colors=['red', 'green', 'blue'])
            
            # Base origin marker
            self.ax.scatter([0], [0], [0], c='black', s=100, marker='s', 
                          edgecolors='black', linewidths=2, label='Base Origin')
            
            # ========================================
            # 2. Draw END-EFFECTOR FRAME using base_T_gripper
            # ========================================
            draw_coordinate_frame(self.ax, base_T_gripper, scale=40.0, label="EE",
                                colors=['darkred', 'darkgreen', 'darkblue'])
            
            # End-effector position marker
            translation = base_T_gripper[:3, 3]
            self.ax.scatter([translation[0]], [translation[1]], [translation[2]], 
                          c='orange', s=120, marker='o', edgecolors='black', 
                          linewidths=2, label='End-Effector')
            
            # ========================================
            # 3. Connection line from base to end-effector
            # ========================================
            self.ax.plot([0, translation[0]], 
                        [0, translation[1]], 
                        [0, translation[2]], 
                        'gray', linestyle='--', linewidth=1.5, alpha=0.6)
            
            # ========================================
            # 4. Configure plot appearance
            # ========================================
            self.ax.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
            self.ax.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
            self.ax.set_zlabel('Z (mm)', fontsize=11, fontweight='bold')
            self.ax.set_title('Dobot Arm Coordinate Frames\nBase (solid) vs End-Effector (dark)', 
                            fontsize=12, fontweight='bold')
            
            # Set fixed axis limits for ~400mm robot workspace
            self.ax.set_xlim([-350, 350])
            self.ax.set_ylim([-350, 350])
            self.ax.set_zlim([0, 350])
            
            # Equal aspect ratio
            self.ax.set_box_aspect([1, 1, 1])
            
            # Legend
            self.ax.legend(loc='upper right', fontsize=9)
            
            # ========================================
            # 5. Display transformation matrix information
            # ========================================
            info = f"""Pose: X={x:.1f}, Y={y:.1f}, Z={z:.1f} mm, R={r:.1f}°
                    Distance: {np.linalg.norm(translation):.1f} mm

                    base_T_gripper (Base → End-Effector):
                    [{base_T_gripper[0,0]:7.4f} {base_T_gripper[0,1]:7.4f} {base_T_gripper[0,2]:7.4f} | {base_T_gripper[0,3]:7.1f}]
                    [{base_T_gripper[1,0]:7.4f} {base_T_gripper[1,1]:7.4f} {base_T_gripper[1,2]:7.4f} | {base_T_gripper[1,3]:7.1f}]
                    [{base_T_gripper[2,0]:7.4f} {base_T_gripper[2,1]:7.4f} {base_T_gripper[2,2]:7.4f} | {base_T_gripper[2,3]:7.1f}]
                    [{base_T_gripper[3,0]:7.4f} {base_T_gripper[3,1]:7.4f} {base_T_gripper[3,2]:7.4f} | {base_T_gripper[3,3]:7.1f}]
                    """
            self.info_text.set_text(info)
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()
        
        return self.ax,
    
    def run(self):
        """Start the animation"""
        ani = FuncAnimation(self.fig, self.update, interval=100, blit=False, cache_frame_data=False)
        plt.show()


def main():
    """Main entry point"""
    try:
        # Connect to Dobot
        port = get_dobot_port()
        print(f"Connecting to Dobot on port: {port}")
        device = Dobot(port=port)
        print("Connected successfully!")
        device.home()
        
        # Get and display initial pose
        initial_pose = device.get_pose()
        print(f"\nInitial pose: X={initial_pose.position.x:.2f}, "
              f"Y={initial_pose.position.y:.2f}, Z={initial_pose.position.z:.2f}, "
              f"R={initial_pose.position.r:.2f}°")
        
        print("\nInitial Transformation Matrix (Base → EE):")
        T_initial = get_robot_arm_matrix(initial_pose)
        print(T_initial)
        print("\nStarting visualization... Close window to exit.")
        print("=" * 70)
        
        # Start visualization
        visualizer = ArmVisualizer(device)
        visualizer.run()
        
        # Cleanup
        device.close()
        print("\nConnection closed.")
        
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
