from configs.environments import environments_config
from ray_utils import kill_environments, create_environments

if __name__ == "__main__":
    kill_environments(environments_config)
    create_environments(environments_config, head=True)
