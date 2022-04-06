import ray
from ray import serve
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from examples.envs_3_7_8.example import Counter_1
from examples.envs_3_9_0.example import Counter

deployments=[{'address': '127.0.0.1:6378', 'namespace': 'serve', 'deployments': [Counter_1], 'fastapi_port':8080},
             {'address': '127.0.0.1:6379', 'namespace': 'serve', 'deployments': [Counter], 'fastapi_port':8081}]

for cluster_info in deployments:
    cli=ray.init(address=cluster_info['address'], namespace=cluster_info['namespace'])
    with cli:
        ray.serve.start(detached=True, http_options={"port": cluster_info['fastapi_port'], "middlewares": [
            Middleware(
                CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
        ]})
        for deployment in cluster_info['deployments']:
            deployment.deploy()
