import ray
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
      # resp=self.IncrResp(**)
      # return resp
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
ray start --head # Start local Ray cluster.
serve start # Start Serve on the local Ray cluster.
ref: https://docs.ray.io/en/latest/serve/deployment.html
dashboard: http://localhost:8265/#/
"""

# This will connect to the running Ray cluster.
ray.init(address="auto", namespace="serve")

Counter.deploy()
