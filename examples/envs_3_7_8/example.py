import ray
from fastapi import FastAPI
from ray import serve
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.openapi.utils import get_openapi
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

def custom_openapi():
    """
    ref: https://fastapi.tiangolo.com/advanced/extending-openapi/
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom title",
        version="2.5.0",
        description="This is a very custom OpenAPI schema",
        routes=app.routes,
    )
    # do changes:
    openapi_schema["paths"]["/incr"]['post']['summary'] = "incrementorrrrr" \
                                             ""
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

@serve.deployment
@serve.ingress(app)
class Counter_1:
  def __init__(self):
      self.count = 0

  class IncrBy(BaseModel):
      by: int = 1
      msg: Optional[str] = Field(None, example="this is a test")

  class IncrResp(BaseModel):
      count: int


  @app.get("/", name="Index", summary="the base API", response_model=IncrResp)
  def get(self):
      return {'count': self.count}

  @app.post("/incr",  summary="the incremonetor")
  def incr(self, incr_by: IncrBy):
      """
      increment the counter
      """
      self.count += incr_by.by
      return {"count": self.count, "message":incr_by.msg}

  @app.get("/decr")
  def decr(self, by: int=1):
      self.count -= by
      return {"count": self.count}

"""
make sure the Ray Cluster is running:
source <path to python 3.7.8 venv/bin/python>
ray start --head  --port 6378 
"""

# This will connect to the running Ray cluster.
ray.init(address='127.0.0.1:6378', namespace="serve_3_7_8")
ray.serve.start(detached=True, http_options={"port": 8080, "middlewares": [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    ]})
Counter_1.deploy()

"""
Notes :-
- use tasks/functions instead of actors/classes, unless statefulness is needed, because out of the 16 (=number of cpus) workers/processes that ray starts,
    the actors own a worker process and tasks do not and tasks are run from the pool and then the worker is reused!
- (NOP) we do not need to transfer serialized objects to maybe we CAN use multiple python versions!
- use SSH tunneling to connect ray in my local and labcas machine!
- create a deployment script in python using popen
- create a cluster using your laptop and labcas and work with multiple clusters in the same python programm
	https://docs.ray.io/en/latest/cluster/ray-client.html
	https://docs.ray.io/en/latest/cluster/cloud.html
- what is the meaning of 0 CPUS?
    - this just serves as a directive for scheduller, that to start this function, it needs there resources,
        but if 0 then it is up to the actor itself how much resources it uses!
- what happens if I have more models than object memory, will the most recent ones be in cache!
- I want to specify that a function needs this many cpus to give directive to a function, but I still want to start many replicas for it!
- each replica runs a single parallel worker
- maybe keep only one actor to put all the large objects and keep all the references! 
- read more what is replicas
- more on async:
    https://medium.com/distributed-computing-with-ray/scaling-python-asyncio-with-ray-aaf42ee03a8e
    https://realpython.com/async-io-python/
    https://docs.ray.io/en/latest/serve/http-servehandle.html
- ray forum:
    https://discuss.ray.io/c/ray-serve/6
- as many number of replicas, so many number of processes!

Todo:
- Implement autoscaling
- write a multi node + pyenv deployment script

Q1. I want to eagerly load all my tensorflow models in the object store using ".put" and let us assume that together they take more memory than available in 
the object store and hence some of them spill to the disk? Is there a better way of loading models in ray serve, which at any time maintains the most recently used models into memory and rest can be spilled?
Q2. Can I do the same with autoscaling for replicas!
    - I can currently achieve this using cpus:0 and then creating the max number of replicas, but this does not make sure it uses autoscaling features properly 
    in ray 

to check dashboard:
http://localhost:8265/
ref: https://docs.ray.io/en/latest/ray-core/ray-dashboard.html
to check memory:
>> ray memory
ref: https://docs.ray.io/en/latest/ray-core/memory-management.html

"""