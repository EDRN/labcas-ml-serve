import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))))
from src.ray_utils import deploy
from configs.basic import environments_config
from configs.environments.environment_A.mappings import deployment_config


if __name__ == "__main__":
    deploy(environments_config, deployment_config)
