from ray_utils import deploy
from configs.environments import environments_config
from configs.env_A import deployment_config, class_name_mappings


if __name__ == "__main__":
    deploy(environments_config, deployment_config, class_name_mappings)
