import ray
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from ray_start import environments_info

def init_deployment(environment_name):

    environment_info = environments_info[environment_name]

    ray.init(address=environment_info['ip'] + ':' + environment_info['port'], namespace=environment_info['namespace'])
    print('resources:', ray.available_resources())
    ray.serve.start(detached=True, http_options={"port": environment_info['serve_port'], "middlewares": [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    ]})

