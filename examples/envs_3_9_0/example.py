import ray
from fastapi import FastAPI
from ray import serve
from pydantic import BaseModel, Field
from typing import Optional
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class Counter:
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

  @app.post("/increase",  summary="the incremonetor")
  def incr(self, incr_by: IncrBy):
      """
      increment the counter
      """
      self.count += incr_by.by
      return {"count": self.count, "message":incr_by.msg}

  @app.get("/decrease")
  def decr(self, by: int=1):
      self.count -= by
      return {"count": self.count}



"""
make sure the Ray Cluster is running:
source <path to python 3.9.0 venv/bin/python>
ray start --head  --port 6379 
"""

# # This will connect to the running Ray cluster.
ray.init(address='127.0.0.1:6379', namespace="serve_3_9_0")
ray.serve.start(detached=True, http_options={"port": 8081, "middlewares": [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    ]})
Counter.deploy()

