from ray import serve
from fastapi import FastAPI
import os
from fastapi.responses import FileResponse
import redis
import sys
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__))))
from src.framework_utils import custom_docs

app = FastAPI()

# edit the api docs, to add various info!
app.openapi = custom_docs(app, "Results", "1.0.0", "Endpoints for tracking and getting results", '/results', [])

output_dir = 'outputs'

redis_url = os.getenv('REDIS_URL')
if redis_url:
    cache = redis.from_url(redis_url)
else:
    cache = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIST_PORT', '6379')), db=0)


def get_task_status(task_id):
    status = cache.hgetall(task_id)
    return status

@serve.ingress(app)
class results:
    def __init__(self):
        pass

    @app.get("/get_results", name="", summary="This endpoint can be used to get results using a task id")
    def get_results(self, task_id: str):
        file_path=os.path.join(output_dir, task_id+'.zip')
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="application/x-zip-compressed", headers={'Content-Disposition': 'attachment; filename='+task_id+'.zip'} )
        else:
            return {'status': 'We do not see the results for this task ID!'}

    @app.get("/task_status", name="", summary="Use this endpoint to get the task status")
    def task_status(self, task_id: str) -> dict:
        status=get_task_status(task_id)
        return {"task_id": task_id, "status": status}
