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
    print("Homing the robotic arm...")
    device.home()
    print("Arm homed. Starting demo moves to three points.")

    # Define three points
    points = [
        (200, 0, 50, 0),
        (250, 50, 50, 90),
        (200, -50, 50, 0),
        (250, 0, 50, -90)
    ]

    for point in points:
        print(f"Moving to point: {point}")
        device.move_to(point[0], point[1], point[2], point[3], wait=True)
        time.sleep(2)  # Pause for a second at each point

        print("Current pose:", device.get_pose())
        time.sleep(1)

    device.close()