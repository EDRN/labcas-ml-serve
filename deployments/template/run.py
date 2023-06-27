from ray import serve
from keras.models import load_model
from fastapi import FastAPI, BackgroundTasks
from skimage.util import img_as_ubyte
from skimage.io import imsave
import uuid
import os, sys
import shutil
import redis
import numpy as np
from fastapi import UploadFile
import aiofiles
import warnings
from skimage.io import imread

from deployments.api_infra.labcas import push_to_labcas_MLOutputs_collection, get_file_metadata_from_labcas
from deployments.api_infra.infra import LabCAS_archive, LabCAS_dataset_path, receive_dir, outputs_dir

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__))))
from src.framework_utils import custom_docs

redis_url = os.getenv('REDIS_URL')
if redis_url:
    cache = redis.from_url(redis_url)
else:
    cache = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIS_PORT', '6379')), db=0)

root_dir='deployments/sam/models'
app = FastAPI()

# edit the api docs, to add various info!
examples=[
    ('/predict', 'post', 0, "(When not uploading an image) provide the id (LabCAS) for the image", []),
    ('/predict', 'post', 1, "(Optional) provide a LabCAS username", []),
]
app.openapi = custom_docs(app, "<Image Predictor> NASA, JPL, LabCAS ML Service (Beta)", "1.0.0", "Docs for <Image Predictor> NASA, JPL", '/sam', examples)

async def eval_images(image_path, model_deplyment_name):
    img=imread(image_path)
    result_im = await serve.get_deployment(model_deplyment_name).get_handle().predict.remote(img)
    return result_im


class model_default:
    def __init__(self):
        self.model = load_model(os.path.join(root_dir, 'unet_256.h5'))

    async def predict(self, img: np.ndarray):
        os.environ["OMP_NUM_THREADS"] = '1'
        p = self.model.predict(img)
        return p

class predict_actor:

    async def predict_(self, task_id, resource, publish_to_labcas, user):

        model_deplyment_name = 'model_default'

        if publish_to_labcas:
            resource_name=os.path.basename(resource)
        else:
            resource_name = resource

        image_path = os.path.join(receive_dir, resource_name)
        image_ext = resource_name.split('.')[-1]
        output_dir = os.path.join(outputs_dir, task_id)
        os.makedirs(output_dir, exist_ok=True)
        cache.hset(task_id, 'status', 'running eval')
        print('running eval')
        result_im = await eval_images(image_path, model_deplyment_name)
        cache.hset(task_id, 'status', 'saving image')
        print('saving result image.')
        result_img_filepath = os.path.join(output_dir, resource_name.replace('.' + image_ext, '_result.' + image_ext))
        imsave(result_img_filepath, img_as_ubyte(result_im), check_contrast=False)

        if publish_to_labcas:

            print('publishing to labcas')
            cache.hset(task_id, 'status', 'publishing results to LabCAS')

            # get metadata from labcas for the target file
            labcas_metadata = get_file_metadata_from_labcas(resource)
            if len(labcas_metadata)==0:
                cache.hset(task_id, 'status', 'Could not retrieve LabCAS information about this file. Exiting.')
                return
            permissions = labcas_metadata.get('OwnerPrincipal', '') # todo: have a fallback permission
            if permissions=='':
                print('WARNING: LabCAS permissions not found!')

            # move the output to a LabCAS dataset
            shutil.move(output_dir, os.path.join(LabCAS_archive, LabCAS_dataset_path))
            # delete the input file
            os.remove(image_path)

            # publish to LabCAS
            push_to_labcas_MLOutputs_collection(task_id, resource, permissions, user=user)
            for out_file_path in os.listdir(os.path.join(LabCAS_archive, LabCAS_dataset_path, task_id)):
                push_to_labcas_MLOutputs_collection(task_id, resource, permissions, filename=os.path.basename(out_file_path), user=user)
        else:
            # zip the results
            cache.hset(task_id, 'status', 'zipping results')
            shutil.make_archive(output_dir, 'zip',
                                root_dir=output_dir)

        cache.hset(task_id, 'status', 'task complete')

@serve.ingress(app)
class sam:
    def __init__(self):
        self.accepted_formats = []

    @app.post("/predict", name="", summary="This does segmentation on an MRI image.")
    async def predict(self, background_tasks: BackgroundTasks, input_image: UploadFile = None, labcas_id: str = '', user: str=''):

        if labcas_id!='':
            # copy the file pointed by labcas_id to the data_dir, so we do not work directly on a file in labcas
            shutil.copy(os.path.join(LabCAS_archive, labcas_id), os.path.join(receive_dir, os.path.basename(labcas_id)))
            resource = labcas_id
            publish_to_labcas = True
        else:
            # ref: https://stackoverflow.com/questions/63580229/how-to-save-uploadfile-in-fastapi
            async with aiofiles.open(os.path.join(receive_dir, input_image.filename), 'wb') as out_file:
                while content := await input_image.read(1024):  # async read chunk
                    await out_file.write(content)  # async write chunk
            resource=input_image.filename
            publish_to_labcas=False

        task_id = str(uuid.uuid4()).replace('-', '')
        serve.get_deployment('predict_actor').get_handle().predict_.remote(self.__class__.__name__, task_id, resource, publish_to_labcas, user)

        if publish_to_labcas:
            results_at = "https://edrn-labcas.jpl.nasa.gov/labcas-ui/d/index.html?dataset_id="+LabCAS_dataset_path+"/" + task_id
        else:
            results_at = "/results/get_results?task_id=" + task_id
        return {"get results at": results_at,
                "check status at": "/results/task_status?task_id="+task_id}




