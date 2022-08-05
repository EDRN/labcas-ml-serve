from fastapi import FastAPI
from ray import serve
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.openapi.utils import get_openapi

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
Notes :-
- use tasks/functions instead of actors/classes, unless statefulness is needed, because out of the 16 (=number of cpus) workers/processes that ray starts,
    the actors own a worker process and tasks do not and tasks are run from the pool and then the worker is reused!
- (NOP) we do not need to transfer serialized objects to maybe we CAN use multiple python versions!
- create a cluster using your laptop and labcas and work with multiple clusters in the same python programm
	https://docs.ray.io/en/latest/cluster/ray-client.html
	https://docs.ray.io/en/latest/cluster/cloud.html
- what is the meaning of 0 CPUS?
    - this just serves as a directive for scheduller, that to start this function, it needs there resources,
        but if 0 then it is up to the actor itself how much resources it uses!
- I want to specify that a function needs this many cpus to give directive to a function, but I still want to start many replicas for it!
- maybe keep only one actor to put all the large objects and keep all the references! 
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
- find out about shared memory:
    - https://stackoverflow.com/questions/66024264/any-python-api-to-get-the-remaining-memory-of-plasma   
    - have multiple sized memories in multiple deployments and see if caching is applied 
    - try the program from the researcher and see if that automatically puts something into shared memory

Q1. I want to eagerly load all my tensorflow models in the object store using ".put" and let us assume that together they take more memory than available in 
the object store and hence some of them spill to the disk? Is there a better way of loading models in ray serve, which at any time maintains the most recently used models into memory and rest can be spilled?
Q2. Can I do the same with autoscaling for replicas!
    - I can currently achieve this using cpus:0 and then creating the max number of replicas, but this does not make sure it uses autoscaling features properly 
    in ray 

increasing the number of replicas based on load can be done by adding another layer for scaling out by updating the config:
https://docs.ray.io/en/releases-1.2.0/serve/advanced.html
autoscaling automatically adds new nodes, if this needs to be done incrementally even for local machines, we need to create fake cluster and fake nodes!

to check dashboard:
http://localhost:8265/
ref: https://docs.ray.io/en/latest/ray-core/ray-dashboard.html
to check memory:
>> ray memory
ref: https://docs.ray.io/en/latest/ray-core/memory-management.html

"""