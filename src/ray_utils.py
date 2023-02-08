import ray
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from ray import serve
import os

def init_get_environment(environment_name, environments_config):

    environment_info = environments_config[environment_name]

    ray.init(address=environment_info['ip'] + ':' + environment_info['port'], namespace=environment_info['namespace'])
    print('resources:', ray.available_resources())
    ray.serve.start(detached=True, http_options={"port": environment_info['serve_port'], "host": "0.0.0.0", "middlewares": [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    ]})


def deploy(environments_config, deployment_config):
    init_get_environment(deployment_config['environment_name'], environments_config)
    for deployment_info in deployment_config['deployments']:
        ray_deployment=serve.deployment(deployment_info['class'],
                                        name=deployment_info['name'],
                                        ray_actor_options={"num_cpus": deployment_info['num_cpus']},
                                        version="1",
                                        num_replicas=deployment_info['num_replicas_base'])

        if deployment_info['name'] == 'auto_scaler':
            ray_deployment.deploy(deployment_config)
        elif 'init' in deployment_config.keys():
            ray_deployment.deploy(deployment_info['init'])
        else:
            ray_deployment.deploy()
    # eg. with ingress: serve.deployment(serve.ingress(app)(Alphan)).options(name='alphan', ray_actor_options={"num_cpus": 0}, num_replicas=1).deploy()

def create_environments(environments_config, head=False):
    for environment_name, environment_info in environments_config.items():
        command = " ray start" +\
                 " --port "+environment_info['port']+\
                 " --object-store-memory "+environment_info['object_store_memory']+\
                 " --dashboard-host 0.0.0.0"+\
                 " --num-cpus "+environment_info['num_cpus']+\
                     (" --head" if head else "")+\
                    "".join([" && python "+deployment for deployment in environment_info['deployments']])

        # Ref: https://unix.stackexchange.com/questions/246813/unable-to-use-source-command-within-python-script
        print('RUNNING on shell:', command)
        os.system(command)


def kill_environments(environments_config):
    for environment_name, environment_info in environments_config.items():
        command = "ray stop"
        print('RUNNING on shell:', command)
        os.system(command)
