import ray
from fastapi import FastAPI
from ray import serve
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import numpy as np
import os
import sys
import logging

root_dir='models/alphan'

def get_logger(log_path):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_path + '.log')
        logger.addHandler(fh)
    return logger

app = FastAPI()

@serve.deployment(num_replicas=5)
def worker(model):
    return model.get_mean()

class MyModel():
    def __init__(self):
        self.np_array=np.zeros((200000, 1000))

    def get_mean(self):
        return np.mean(self.np_array)

@serve.deployment(num_replicas=2)
@serve.ingress(app)
class Orchestrator:
    def __init__(self):
        self.count = 0
        self.logger=get_logger(root_dir+'memory_test')
        self.model=ray.put(MyModel()) # MyModel()

    @app.get("/predict")
    async def predict(self):
        result = await worker.get_handle().remote(self.model)
        return {'result': str(result)}

"""
make sure the Ray Cluster is running:
source <path to python 3.7.8 venv/bin/python>
ray start --head  --port 6378 
"""

# This will connect to the running Ray cluster.
ray.init(address='127.0.0.1:6378', namespace="serve_3_7_8")
print('resources:', ray.available_resources())
ray.serve.start(detached=True, http_options={"port": 8080, "middlewares": [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    ]})

worker.deploy()
Orchestrator.deploy()


