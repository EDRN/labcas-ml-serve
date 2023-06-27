from ray import serve
from fastapi import FastAPI, BackgroundTasks
import uuid
import os, sys
import shutil
import redis
from fastapi import UploadFile
import aiofiles
import warnings
import numpy as np
import torch
import cv2
import supervision as sv
import pydicom
import matplotlib.image
import requests
from tqdm import tqdm

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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

def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024 * 8), total=int(int(r.headers['Content-length'])/float(1024 * 8))):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

def increase_contrast(image):
    minval = np.percentile(image, 50)
    maxval = np.percentile(image, 98)
    image = np.clip(image, minval, maxval)
    image = (((image - minval) / (maxval - minval)) * 255).astype(np.uint8)
    image=np.stack((image,) * 3, axis=-1)
    return image
async def eval_images(image_path, model_deplyment_name):
    # ref: pip install 'git+https://github.com/facebookresearch/segment-anything.git'
    # model at: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # get https://media.roboflow.com/notebooks/examples/dog.jpeg
    if '.dcm' in image_path.lower():
        ds = pydicom.dcmread(image_path)
        if hasattr(ds, 'pixel_array'):
            image = ds.pixel_array
            if len(image.shape) == 3:
                image = image[:, :, 0]
            image = increase_contrast(image)
        else:
            print('The provided DICOM image does not have a valid image array!')
            return None
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blank_image = np.full(image.shape, 255, dtype=int)
    sam_result = await serve.get_deployment(model_deplyment_name).get_handle().predict.remote(image)
    mask_annotator = sv.MaskAnnotator()
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotations = mask_annotator.annotate(scene=blank_image, detections=detections).astype(np.uint8)
    return annotated_image, annotations


class sam_default:
    def __init__(self):
        if not os.path.exists(os.path.join(root_dir, 'sam_vit_b_01ec64.pth')):
            download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                 dest_folder=root_dir)
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"
        CHECKPOINT_PATH = os.path.join(root_dir, "sam_vit_h_4b8939.pth")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)
        self.model = mask_generator

    async def predict(self, img: np.ndarray):
        # os.environ["OMP_NUM_THREADS"] = '1'
        print('predicting now...')
        result = self.model.generate(img)
        print('prediction complete!')
        return result

class sam_predict_actor:

    async def predict_(self, task_id, resource, publish_to_labcas, user):

        model_deplyment_name = 'sam_default'

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
        annotated_image, annotations = await eval_images(image_path, model_deplyment_name)
        cache.hset(task_id, 'status', 'saving image')
        result_img_filepath = os.path.join(output_dir, resource_name.replace('.' + image_ext, '_result.png'))
        result_mask = os.path.join(output_dir, resource_name.replace('.' + image_ext, '_mask.png'))
        result_mask_npy = os.path.join(output_dir, resource_name.replace('.' + image_ext, '_mask.npy'))
        matplotlib.image.imsave(result_img_filepath, annotated_image)
        matplotlib.image.imsave(result_mask, annotations)
        with open(result_mask_npy, 'wb') as f:
            np.save(f, annotations)
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
        serve.get_deployment('sam_predict_actor').get_handle().predict_.remote(task_id, resource, publish_to_labcas, user)

        if publish_to_labcas:
            results_at = "https://edrn-labcas.jpl.nasa.gov/labcas-ui/d/index.html?dataset_id="+LabCAS_dataset_path+"/" + task_id
        else:
            results_at = "/results/get_results?task_id=" + task_id
        return {"get results at": results_at,
                "check status at": "/results/task_status?task_id="+task_id}




