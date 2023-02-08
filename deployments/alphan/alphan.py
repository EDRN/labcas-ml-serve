from ray import serve
from keras.models import load_model
import logging
from fastapi import FastAPI, BackgroundTasks
from skimage.util import img_as_ubyte
from skimage.io import imsave
import uuid
import os, sys
import shutil
import redis
import numpy as np
from skimage.exposure import rescale_intensity, equalize_adapthist, adjust_sigmoid
from skimage.io import imread
from skimage.util import img_as_bool, view_as_windows
from fastapi import UploadFile
import aiofiles
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__))))
from utils import extract_regionprops, bw_watershed, pad_to_n, plot_contours
from src.framework_utils import custom_docs
from deployments.api_infra.labcas import push_to_labcas_MLOutputs_collection, get_file_metadata_from_labcas
from deployments.api_infra.infra import LabCAS_archive, LabCAS_dataset_path, receive_dir, outputs_dir

# Todo: store the redis ports and root paths in a config file!
cache = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIST_PORT', '6379')), db=0)

root_dir='deployments/alphan/models'
app = FastAPI()

# edit the api docs, to add various info!
examples=[
    ('/predict', 'post', 0, "Provide the name of the model to be used", ['unet_default']),
    ('/predict', 'post', 1, "Do you want to extract the region properties", ['True', 'False']),
    ('/predict', 'post', 2, "Choose the image processing window size", [128, 64, 256]),
    ('/predict', 'post', 3, "(When not uploading an image) provide the id (LabCAS) for the image", []),
    ('/predict', 'post', 4, "(Optional) provide a LabCAS username", []),
]
app.openapi = custom_docs(app, "Nuclei Position Detector by Alphan Altinok NASA, JPL, LabCAS ML Service (Beta)", "1.0.0", "Docs for Nuclei Position Detector by Alphan Altinok NASA, JPL", '/alphan', examples)

# todo: Fix the loggers paths and put it into common utils!
def get_logger(log_path):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_path + '.log')
        logger.addHandler(fh)
    return logger


async def eval_images(image_path, model_deplyment_name="unet", w=64):

    # store image shape before padding
    print('reading image')
    im=imread(image_path)
    sh = im.shape

    print('rescaling intensity')
    im = rescale_intensity(1.0 * im)

    print('padding')
    im = pad_to_n(im, w=w)

    print('tiling the images')
    imw = view_as_windows(im, (w,w), (w,w))
    bw = np.zeros_like(im)
    imb = view_as_windows(bw, (w,w), (w,w))

    print('running pre-processing and predictions')
    for i in range(imw.shape[0]):
        for j in range(imw.shape[1]):
            img = np.expand_dims(imw[i,j,...], axis=[0,3])
            print('img.shape:', img.shape)
            imw = await serve.get_deployment('preprocessing').get_handle().preprocess.remote(img)
            print('imw.shape:', imw.shape)
            # Todo: maybe move this somewhere else!
            p = await serve.get_deployment(model_deplyment_name).get_handle().predict.remote(imw)
            p = p[0,:,:,0]
            b = p > 0.5
            imb[i,j,...] = b

    print('stitching together')
    # revert back to original image shape
    im = im[:sh[0],:sh[1]]
    bw = bw[:sh[0],:sh[1]]
    print('converting bw image to boolean')
    bw = img_as_bool(bw)

    print('running watershed post-processing')
    # postprocess
    bw = bw_watershed(bw)

    return bw, im

class preprocessing:
    async def preprocess(self, imw: np.ndarray):
        imw = equalize_adapthist(imw)
        imw = adjust_sigmoid(imw)
        return imw

class unet:
    def __init__(self):
        self.model = load_model(os.path.join(root_dir, 'unet_default.h5'))
        self.logger = get_logger(root_dir + 'Unet')
        self.logger.info('model loaded')

    async def predict(self, img: np.ndarray):
        os.environ["OMP_NUM_THREADS"] = '1'
        p = self.model.predict(img)
        return p

class predict_actor:
    async def predict_(self, class_name, task_id, resource, model_name, is_extract_regionprops,
                          window, publish_to_labcas, user):
        if model_name=='unet_default':
            model_deplyment_name='unet'
        else:
            cache.hset(task_id, 'status', 'Error: the requested model has not been deployed yet!')
            return

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
        bw, im = await eval_images(image_path, model_deplyment_name, w=window)

        bw_filepath = os.path.join(output_dir, resource_name.replace('.' + image_ext, '_bw.' + image_ext))
        contour_filpath = bw_filepath.replace('_bw', '_ov')
        print('saving BW image.')
        imsave(bw_filepath, img_as_ubyte(bw), check_contrast=False)
        cache.hset(task_id, 'status', 'plotting contours')
        print('plotting contours')
        plot_contours(bw, im, contour_filpath)

        if is_extract_regionprops == 'True':
            cache.hset(task_id, 'status', 'extracting region properties')
            print('extracting region props')
            image_df = extract_regionprops(image_path, bw_filepath)
            image_df.to_csv(bw_filepath.replace('.' + image_ext, '.csv'))


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
            push_to_labcas_MLOutputs_collection(task_id, resource_name, permissions, user=user)
            for out_file_path in os.listdir(os.path.join(LabCAS_archive, LabCAS_dataset_path, task_id)):
                push_to_labcas_MLOutputs_collection(task_id, resource_name, permissions, filename=os.path.basename(out_file_path), user=user)
        else:
            # zip the results
            cache.hset(task_id, 'status', 'zipping results')
            shutil.make_archive(output_dir, 'zip',
                                root_dir=output_dir)

        cache.hset(task_id, 'status', 'task complete')

@serve.ingress(app)
class alphan:
    def __init__(self):
        self.accepted_formats = ['png', 'tif', 'tiff']
        self.logger=get_logger(root_dir+'Alphan')
        self.logger.info('models loaded')

    @app.post("/predict", name="", summary="This endpoint detects nuclei positions in an image.")
    async def predict(self, background_tasks: BackgroundTasks, input_image: UploadFile = None, model_name: str = 'unet_default', is_extract_regionprops: str = 'True',
                      window: int=128, labcas_id: str = '', user: str=''):

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
        # background_tasks.add_task(predict_, self.__class__.__name__, task_id, resource, model_name, is_extract_regionprops,
        #         window, publish_to_labcas, user)

        # FAST API background tasks don't work here as expected. So using a Ray actor to do run it as a background task!
        # ref: https://github.com/ray-project/ray/issues/24627
        # actor = AsyncActor.remote()
        # actor.predict_.remote(self.__class__.__name__, task_id, resource, model_name, is_extract_regionprops, window, publish_to_labcas, user)

        # okay, so the above method works but gived some pickle thread lock error when setting status through Redis object!!!
        # So, guess we will have to have a dedicated deployment  after all. The only difference is that an actor gets created and then dies when the job is over.
        serve.get_deployment('predict_actor').get_handle().predict_.remote(self.__class__.__name__, task_id, resource, model_name, is_extract_regionprops, window, publish_to_labcas, user)

        if publish_to_labcas:
            # TODO: make the url to labcas as a congig param
            results_at = "https://edrn-labcas.jpl.nasa.gov/labcas-ui/d/index.html?dataset_id="+LabCAS_dataset_path+"/" + task_id
        else:
            results_at = "/results/get_results?task_id=" + task_id
        return {"get results at": results_at,
                "check status at": "/results/task_status?task_id="+task_id}


    @app.get("/train", name="", summary="This endpoint trains a model to detect nuclei positions in an image.")
    async def train(self):
        return None


## OLD CODE with preprocessing not deployed separately: will keep this here as ref. for now!!
# async def eval_images(image_path, model_deplyment_name="unet", w=64):
#
#     print('rescaling intensity')
#     im = rescale_intensity(1.0*imread(image_path))
#     print('adjusting sigmoid')
#     im = adjust_sigmoid(equalize_adapthist(im))
#
#     # store image shape before padding
#     sh = im.shape
#     print('padding')
#     im = pad_to_n(im, w=w)
#
#     print('tiling the images')
#     bw = np.zeros_like(im)
#     imw = view_as_windows(im, (w,w), (w,w))
#     imb = view_as_windows(bw, (w,w), (w,w))
#
#     print('running predictions')
#     for i in range(imb.shape[0]):
#         for j in range(imb.shape[1]):
#             img = np.expand_dims(imw[i,j,...], axis=[0,3])
#             # Todo: maybe move this somewhere else!
#             p = await serve.get_deployment(model_deplyment_name).get_handle().predict.remote(img)
#             p = p[0,:,:,0]
#             b = p > 0.5
#             imb[i,j,...] = b
#
#     print('stitching together')
#     # revert back to original image shape
#     im = im[:sh[0],:sh[1]]
#     bw = bw[:sh[0],:sh[1]]
#     bw = img_as_bool(bw)
#
#     print('running watershed postprocessing')
#     # postprocess
#     bw = bw_watershed(bw)
#
#     return bw, im