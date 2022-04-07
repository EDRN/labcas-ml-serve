from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

import ray
from ray import serve
from example import Counter
from ray_start import deployments_info


deployment_name='example_B'
deployment_info=deployments_info[deployment_name]

ray.init(address=deployment_info['ip']+':'+deployment_info['port'], namespace=deployment_info['namespace'])
print('resources:', ray.available_resources())
ray.serve.start(detached=True, http_options={"port": deployment_info['serve_port'], "middlewares": [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    ]})

Counter.deploy()