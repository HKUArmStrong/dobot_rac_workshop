import os
import yaml
from pydobotplus import Dobot


def get_dobot_port():
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]

if __name__ == "__main__":
    """
    Connect to the Dobot and print the current pose, then exit.
    """
    port = get_dobot_port()
    device = Dobot(port=port)
    try:
        print(device.get_pose(), flush=True)
    finally:
        device.close()