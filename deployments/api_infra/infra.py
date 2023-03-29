from ray import serve
from fastapi import FastAPI
import os
from fastapi.responses import FileResponse
import redis
import sys
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__))))
from src.framework_utils import custom_docs

## ============ File path configurations

LabCAS_archive = '/labcas-data/labcas-backend/archive/edrn' # this is the docker internal path (and if changed then it needs to be changed in the docker-compose)
real_LabCAS_archive='/labcas-data/labcas-backend/archive/edrn' # the actual path to the labcas archive (not the docker internal one!)
LabCAS_dataset_path = 'MLOutputs/Outputs' # the dataset in labcas where the results are saved when the model triggered from labcas UI
data_dir = os.path.join(LabCAS_archive, 'ML_data_dir') # this is a designated dir within the labcas archive to do various temporary stuff
receive_dir = os.path.join(data_dir, 'received_data') # this is where we receive data from user or copy to internally from labcas to be used by the model
outputs_dir = os.path.join(data_dir, 'output') # this is where the outputs are saved. But when triggered from within labcas, the output is moved to the LabCAS_dataset_path dataset
dummy_data='/usr/src/app/deployments/api_infra/dummy_data/test_image_with_cells.png'

os.makedirs(receive_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# ========= LabCAS Solr details

solr_url = os.getenv('SOLR_URL', 'http://localhost:8983/solr')


# ==== Fast API object
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
        file_path=os.path.join(outputs_dir, task_id+'.zip')
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="application/x-zip-compressed", headers={'Content-Disposition': 'attachment; filename='+task_id+'.zip'} )
        else:
            return {'status': 'We do not see the results for this task ID!'}

    @app.get("/task_status", name="", summary="Use this endpoint to get the task status")
    def task_status(self, task_id: str) -> dict:
        status=get_task_status(task_id)
        return {"task_id": task_id, "status": status}

