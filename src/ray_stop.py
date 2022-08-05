import sys
import os
sys.path.insert(0, os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))
from configs.basic import environments_config
from ray_utils import kill_environments

if __name__ == "__main__":
    kill_environments(environments_config)

