import ray
from fastapi import FastAPI
from ray import serve
import os
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import time
import logging
import datetime

app = FastAPI()
root_dir='models/alphan'

def get_logger(log_path):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_path + '.log')
        logger.addHandler(fh)
    return logger

@serve.deployment(ray_actor_options={"num_cpus": 0}, num_replicas=5)
def the_doer():
    time.sleep(10)
    return 1

@serve.deployment(ray_actor_options={"num_cpus": 0}, _autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_num_ongoing_requests_per_replica": 1,
    },
    version="v1")

def the_doer_2():
    time.sleep(10)
    return 1

@serve.deployment(ray_actor_options={"num_cpus": 0}, num_replicas=5)
def the_doer_3():
    time.sleep(10)
    return 1

@serve.deployment(ray_actor_options={"num_cpus": 0}, num_replicas=5)
def the_doer_4():
    time.sleep(10)
    return 1

@serve.deployment(ray_actor_options={"num_cpus":0}, num_replicas=5)
def the_doer_5():
    time.sleep(10)
    return 1


@serve.deployment(ray_actor_options={"num_cpus": 0})
@serve.ingress(app)
class Alphan:
    def __init__(self):
        self.accepted_formats = ['png', 'tif', 'tiff']
        self.logger=get_logger(root_dir)
        self.logger.info('model loaded')
        self.doer=the_doer.get_handle()


    @app.get("/predict", name="", summary="This endpoint predicts neuclei boundries in an image.")
    async def predict(self):
        self.logger.info('1')
        start=datetime.datetime.now()
        r = await self.doer.remote()
        self.logger.info('2- ' + str(datetime.datetime.now()-start))
        return {"status": r}

if __name__ == "__main__":
    cli=ray.init(address='127.0.0.1:6378', namespace="serve")
    ray.serve.start(detached=True, http_options={"port": 8080, "middlewares": [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    ]})
    the_doer.deploy()
    the_doer_2.deploy()
    the_doer_3.deploy()
    the_doer_4.deploy()
    the_doer_5.deploy()
    Alphan.deploy()

