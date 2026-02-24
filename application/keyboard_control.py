#!/usr/bin/env python3
"""
Keyboard Control for Dobot Arm
Controls:
    - Arrow Keys / WASD: Move X/Y position
    - Q/E: Move Z up/down
    - Z/C: Rotate counter-clockwise/clockwise
    - H: Home the arm
    - P: Print current pose
    - Space: Stop/Reset to current position
    - ESC: Exit
"""

import os
import yaml
import time
import sys
from pydobotplus import Dobot
from pydobotplus.dobotplus import MODE_PTP
from pynput import keyboard


class KeyboardArmController:
    def __init__(self, port):
        self.device = Dobot(port=port)
        self.step_size = 5  # mm for X, Y, Z movement
        self.rotation_step = 5  # degrees for rotation
        self.running = True
        self.gripper_closed = False  # Track gripper state
        
        # Initialize position
        self.current_pose = self.device.get_pose()
        print(f"Starting pose: {self.current_pose}")
        print("\n" + "="*60)
        print("KEYBOARD CONTROL ACTIVE")
        print("="*60)
        print("Controls:")
        print("  Arrow Keys / WASD : Move X/Y")
        print("  Q / E             : Move Z up/down")
        print("  Z / C             : Rotate counter-clockwise/clockwise")
        print("  H                 : Home the arm (clears alarms)")
        print("  P                 : Print current pose")
        print("  Space             : Toggle gripper (close/open)")
        print("  +/-               : Increase/decrease step size")
        print("  ?                 : Show controls (help)")
        print("  ESC               : Exit")
        print("="*60 + "\n")
        
    def move_relative(self, dx=0, dy=0, dz=0, dr=0):
        """Move the arm relative to current position"""
        try:
            pose = self.device.get_pose()
            x, y, z, r = pose.position.x, pose.position.y, pose.position.z, pose.position.r
            
            new_x = x + dx
            new_y = y + dy
            new_z = z + dz
            new_r = r + dr
            
            # Safety bounds (adjust as needed for your workspace)
            new_x = max(-300, min(300, new_x))
            new_y = max(-300, min(300, new_y))
            new_z = max(-50, min(150, new_z))
            
            self.device.move_to(new_x, new_y, new_z, new_r, wait=False)
            print(f"Moving to: X={new_x:.1f}, Y={new_y:.1f}, Z={new_z:.1f}, R={new_r:.1f}")
        except Exception as e:
            print(f"Error moving arm: {e}")
    
    def home(self):
        """Home the arm and clear alarms"""
        print("Clearing alarms...")
        self.device.clear_alarms()
        print("Moving to safe position (200, 0, 50)...")
        move_idx = self.device.move_to(200, 0, 50, 0, mode=MODE_PTP.MOVJ_XYZ, wait=False)
        self.device.wait_for_cmd(move_idx)
        print("Homing arm...")
        self.device.home()
        print("Homing complete")
    
    def show_help(self):
        """Display control keys"""
        print("\n" + "="*60)
        print("CONTROL KEYS")
        print("="*60)
        print("Movement:")
        print("  Arrow Keys / WASD : Move X/Y position")
        print("  Q / E             : Move Z up/down")
        print("  Z / C             : Rotate counter-clockwise/clockwise")
        print("\nCommands:")
        print("  H                 : Home the arm (clears alarms)")
        print("  P                 : Print current pose")
        print("  Space             : Toggle gripper (close/open)")
        print("  +/-               : Increase/decrease step size")
        print("  ?                 : Show this help")
        print("  ESC               : Exit")
        print("\nCurrent Settings:")
        print(f"  Step size         : {self.step_size}mm")
        print(f"  Rotation step     : {self.rotation_step}°")
        print(f"  Gripper state     : {'Closed' if self.gripper_closed else 'Open'}")
        print("="*60 + "\n")
        
    def print_pose(self):
        """Print current pose"""
        pose = self.device.get_pose()
        print(f"Current pose: X={pose.position.x:.2f}, Y={pose.position.y:.2f}, "
              f"Z={pose.position.z:.2f}, R={pose.position.r:.2f}")
    
    def toggle_gripper(self):
        """Toggle gripper state - close or open"""
        self.gripper_closed = not self.gripper_closed
        if self.gripper_closed:
            print("Closing gripper...")
            self.device.grip(True)
        else:
            print("Opening gripper...")
            self.device.grip(False)
        print(f"Gripper is now: {'Closed' if self.gripper_closed else 'Open'}")
        
    def on_press(self, key):
        """Handle key press events"""
        try:
            # Handle character keys
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                
                # X/Y Movement (WASD)
                if char == 'w':
                    self.move_relative(dx=self.step_size)
                elif char == 's':
                    self.move_relative(dx=-self.step_size)
                elif char == 'a':
                    self.move_relative(dy=self.step_size)
                elif char == 'd':
                    self.move_relative(dy=-self.step_size)
                
                # Z Movement
                elif char == 'q':
                    self.move_relative(dz=self.step_size)
                elif char == 'e':
                    self.move_relative(dz=-self.step_size)
                
                # Rotation
                elif char == 'z':
                    self.move_relative(dr=self.rotation_step)
                elif char == 'c':
                    self.move_relative(dr=-self.rotation_step)
                
                # Commands
                elif char == 'h':
                    self.home()
                elif char == 'p':
                    self.print_pose()
                elif char == '?':
                    self.show_help()
                
                # Step size adjustment
                elif char == '+' or char == '=':
                    self.step_size += 1
                    print(f"Step size increased to {self.step_size}mm")
                elif char == '-':
                    self.step_size = max(1, self.step_size - 1)
                    print(f"Step size decreased to {self.step_size}mm")
                    
            # Handle special keys
            else:
                # Arrow keys
                if key == keyboard.Key.up:
                    self.move_relative(dx=self.step_size)
                elif key == keyboard.Key.down:
                    self.move_relative(dx=-self.step_size)
                elif key == keyboard.Key.left:
                    self.move_relative(dy=self.step_size)
                elif key == keyboard.Key.right:
                    self.move_relative(dy=-self.step_size)
                
                # Space to toggle gripper
                elif key == keyboard.Key.space:
                    self.toggle_gripper()
                
                # ESC to exit
                elif key == keyboard.Key.esc:
                    print("\nExiting...")
                    self.running = False
                    return False
                    
        except Exception as e:
            print(f"Error handling key: {e}")
    
    def run(self):
        """Start the keyboard listener"""
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()
        
        # Cleanup
        print("Closing connection...")
        self.device.close()
        print("Done!")


def get_dobot_port():
    """Load Dobot port from config file"""
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]


if __name__ == "__main__":
    try:
        port = get_dobot_port()
        controller = KeyboardArmController(port)
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
