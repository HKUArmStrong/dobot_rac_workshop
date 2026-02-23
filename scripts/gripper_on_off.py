import os
import yaml
import time
from pydobotplus import Dobot


def get_dobot_port():
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]

if __name__ == "__main__":
    port = get_dobot_port()
    device = Dobot(port=port)
    print("Starting gripper toggle demo. Press Ctrl-C to stop.")
    
    try:
        while True:
            print("Gripper ON (closing)")
            device.grip(True)
            time.sleep(1)
            
            print("Gripper OFF (opening)")
            device.grip(False)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping gripper toggle.")
    finally:
        device.grip(False)  # Ensure gripper is open before closing
        device.close()