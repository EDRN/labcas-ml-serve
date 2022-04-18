from ray import serve
import os
import logging

def get_logger(log_path):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_path + '.log')
        logger.addHandler(fh)
    return logger

@serve.deployment(name='autoscaler', ray_actor_options={"num_cpus": 0}, num_replicas=1)
class Auto_Scaler:
    def __init__(self):
        self.current_requests = 0
        self.logger=get_logger('auto_scale')

    def update_current_requests(self, deployment_name, num_requests):
        self.current_requests+=num_requests
        self.logger.info('requests now:', self.current_requests)
        # TODO: here put the logic to evaluate if there is a need to increase the number of replicas

if __name__ == "__main__":
    from ray_utils import init_deployment
    init_deployment('environment_A')
    Auto_Scaler.deploy()

