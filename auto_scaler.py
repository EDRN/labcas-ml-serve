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

root_dir='models'

class Auto_Scaler:
    def __init__(self):
        self.current_requests = {}
        self.logger=get_logger(os.path.join(root_dir, 'auto_scale'))

    def update_current_requests(self, deployment_name):
        if deployment_name not in self.current_requests.keys():
            self.current_requests[deployment_name]=0
        self.current_requests[deployment_name]+=1
        self.logger.info('requests:'+str(self.current_requests))
        if self.current_requests[deployment_name]>6:
            num_replicas=4
            if serve.get_deployment(deployment_name).num_replicas<num_replicas:
                self.logger.info('redeploying '+deployment_name + ' to num replicas: '+str(num_replicas))
                serve.get_deployment(deployment_name).options(num_replicas=num_replicas).deploy()

if __name__ == "__main__":
    from ray_utils import init_deployment
    init_deployment('environment_A')
    serve.deployment(Auto_Scaler).options(name='autoscaler', ray_actor_options={"num_cpus": 0}, num_replicas=1).deploy()

