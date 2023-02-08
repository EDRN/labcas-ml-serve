from ray import serve
import os
import logging
import math

## RETIRED (as Ray provides this functionality natively now): this was a custom autoscaler deployment
# IT could still be useful to look at this pattern of a supervisor deployer.

def get_logger(log_path):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_path + '.log')
        logger.addHandler(fh)
    return logger

root_dir='models'

class auto_scaler:
    def __init__(self, deployment_config):
        # convert the deployment_config into a lookable format by the deployment name
        self.deployment_config = {v['name']: v for v in deployment_config['deployments']}
        self.current_requests = {} # todo: instead of this, use rate
        self.logger=get_logger(os.path.join(root_dir, 'auto_scale'))
        self.current_replicas={}

    def update_current_requests(self, deployment_name):
        if deployment_name not in self.current_requests.keys():
            self.current_requests[deployment_name]=0
        self.current_requests[deployment_name]+=1
        self.logger.info('requests:'+str(self.current_requests))
        # todo: the below will now represent the current replicas once we redeploy, manually keep track
        current_replicas=serve.get_deployment(deployment_name).num_replicas
        if current_replicas < self.deployment_config[deployment_name]['max_replicas']:
            # calculate how many replicas we should have right now
            target_replicas=math.ceil(float(self.current_requests[deployment_name])/self.deployment_config[deployment_name]['target_num_ongoing_requests_per_replica'])
            if target_replicas > current_replicas:
                self.logger.info('redeploying '+deployment_name + ' to more replicas: '+str(current_replicas)+'-->'+str(target_replicas))
            serve.get_deployment(deployment_name).options(num_replicas=target_replicas).deploy()

            # todo: write down the code for scale down
