from ray import serve
from fastapi import FastAPI, BackgroundTasks
import uuid
import os, sys
import shutil
import redis
from fastapi import UploadFile
import aiofiles
import warnings
from importlib import import_module
import numpy as np
import torch
from scipy.ndimage import zoom
from einops import repeat
import matplotlib.image
import pydicom
import urllib.request

from deployments.api_infra.labcas import push_to_labcas_MLOutputs_collection, get_file_metadata_from_labcas
from deployments.api_infra.infra import LabCAS_archive, LabCAS_dataset_path, receive_dir, outputs_dir

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__))))
from segment_anything_sv import sam_model_registry as sam_model_registry_local

from src.framework_utils import custom_docs

redis_url = os.getenv('REDIS_URL')
if redis_url:
    cache = redis.from_url(redis_url)
else:
    cache = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIS_PORT', '6379')), db=0)

root_dir='deployments/samed/models'
app = FastAPI()

# edit the api docs, to add various info!
examples=[
    ('/predict', 'post', 0, "(When not uploading an image) provide the id (LabCAS) for the image", []),
    ('/predict', 'post', 1, "(Optional) provide a LabCAS username", []),
]
app.openapi = custom_docs(app, "<Image Predictor> NASA, JPL, LabCAS ML Service (Beta)", "1.0.0", "Docs for <Image Predictor> NASA, JPL", '/samed', examples)


def increase_contrast(image):
    minval = np.percentile(image, 50)
    maxval = np.percentile(image, 98)
    # todo: try histogram eqalization
    image = np.clip(image, minval, maxval)
    image = (((image - minval) / (maxval - minval)) * 255).astype(np.uint8)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image
async def eval_images(image_path, model_deplyment_name):

    # ref: https://github.com/hitachinsk/SAMed/tree/main

    img_size = 512
    patch_size = [img_size, img_size]
    ds = pydicom.dcmread(image_path)
    if hasattr(ds, 'pixel_array'):
        image=ds.pixel_array
        if len(image.shape) == 3:
            image = image[:, :, 0]
        image=increase_contrast(image)
    else:
        print('The provided DICOM image does not have a valid image array!')
        return None
    x, y = image.shape[0], image.shape[1]
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
    inputs = torch.from_numpy(image).unsqueeze(
        0).unsqueeze(0).float()
    inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
    with torch.no_grad():
        outputs = await serve.get_deployment(model_deplyment_name).get_handle().predict.remote(inputs)
        output_masks = outputs['masks']
        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        pred = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            pred = zoom(pred, (x / patch_size[0], y / patch_size[1]), order=0)
    return pred


class samed_default:
    def __init__(self):
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                                   "sam_vit_b_01ec64.pth")
        self.class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach',
                         7: 'aorta', 8: 'pancreas'}
        num_classes = 8
        img_size = 512
        self.patch_size = [img_size, img_size]
        rank = 4
        ckpt = os.path.join(root_dir, 'sam_vit_b_01ec64.pth')
        lora_ckpt = os.path.join(root_dir, 'epoch_159.pth')
        vit_name = 'vit_b'
        sam, img_embedding_size = sam_model_registry_local[vit_name](image_size=img_size,
                                                                     num_classes=num_classes,
                                                                     checkpoint=ckpt, pixel_mean=[0, 0, 0],
                                                                     pixel_std=[1, 1, 1])
        pkg = import_module('sam_lora_image_encoder')
        model = pkg.LoRA_Sam(sam, rank)
        model.load_lora_parameters(lora_ckpt)
        self.model = model

    async def predict(self, img: np.ndarray):
        os.environ["OMP_NUM_THREADS"] = '1'
        multimask_output = True
        print('predicting now...')
        outputs = self.model(img, multimask_output, self.patch_size[0])
        print('prediction complete!')
        return outputs

class samed_predict_actor:

    async def predict_(self, task_id, resource, publish_to_labcas, user):

        model_deplyment_name = 'samed_default'

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
        if result_im is None:
            # a valid array was not found in the DICOM file
            return
        cache.hset(task_id, 'status', 'saving image')
        result_mask = os.path.join(output_dir, resource_name.replace('.' + image_ext, '_mask.png'))
        matplotlib.image.imsave(result_mask, result_im)
        result_mask_npy = os.path.join(output_dir, resource_name.replace('.' + image_ext, '_mask.npy'))
        with open(result_mask_npy, 'wb') as f:
            np.save(f, result_im)
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
class samed:
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
        serve.get_deployment('samed_predict_actor').get_handle().predict_.remote(task_id, resource, publish_to_labcas, user)

        if publish_to_labcas:
            results_at = "https://edrn-labcas.jpl.nasa.gov/labcas-ui/d/index.html?dataset_id="+LabCAS_dataset_path+"/" + task_id
        else:
            results_at = "/results/get_results?task_id=" + task_id
        return {"get results at": results_at,
                "check status at": "/results/task_status?task_id="+task_id}
