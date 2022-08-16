from ray import serve
from fastapi.openapi.utils import get_openapi
from keras.models import load_model
import logging
from fastapi import FastAPI, BackgroundTasks
from skimage.util import img_as_ubyte
from skimage.io import imsave
import uuid
import os
import shutil
import redis
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist, adjust_sigmoid
from skimage.feature import peak_local_max
from skimage.io import imread
from skimage.measure import label, find_contours, regionprops_table
from skimage.morphology import disk, remove_small_objects, opening, remove_small_holes
from skimage.segmentation import watershed, expand_labels
from skimage.util import img_as_bool, view_as_windows
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import asyncio

# Todo: store the redis ports and root paths in a config file!

cache = redis.Redis(host='localhost', port=6379, db=0)

app = FastAPI()

# Todo: fix the paths in the api docs so that things could be executed from there!
def custom_openapi():
    """
    ref: https://fastapi.tiangolo.com/advanced/extending-openapi/
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Alphan's LABCAS Predictor",
        version="1.0.0",
        description="This is the docs for Alphan's LABCAS predictor",
        routes=app.routes,
    )

    # START: ====== predict endpoint info

    # param 1 info
    openapi_schema["paths"]["/predict"]['get']['parameters'][0]['description'] = "Give the image name"
    openapi_schema["paths"]["/predict"]['get']['parameters'][0]['examples'] = {
        'example1': {'value': '8_1_0_0.png'}
    }

    # param 2 info
    openapi_schema["paths"]["/predict"]['get']['parameters'][1][
        'description'] = "Provide the name of the model to be used."
    openapi_schema["paths"]["/predict"]['get']['parameters'][1]['examples'] = {
        'example1': {'model_name': 'unet_default'}
    }

    # param 3 info
    openapi_schema["paths"]["/predict"]['get']['parameters'][2][
        'description'] = "Do you want to extract the region properties"
    openapi_schema["paths"]["/predict"]['get']['parameters'][2]['examples'] = {
        'example1': {'is_extract_regionprops': 'True'},
        'example2': {'is_extract_regionprops': 'False'}
    }

    # param 4 info
    openapi_schema["paths"]["/predict"]['get']['parameters'][3][
        'description'] = "Choose the image processing window size"
    openapi_schema["paths"]["/predict"]['get']['parameters'][3]['examples'] = {
        'example2': {'window': 128},
        'example1': {'window': 64},
        'example3': {'window': 256}
    }

    # END: ====== predict endpoint info


    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# todo: Fix the loggers paths!
def get_logger(log_path):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_path + '.log')
        logger.addHandler(fh)
    return logger

root_dir='deployments/alphan/models'

async def eval_images(image_path, model_deplyment_name="unet", w=64):

    im = rescale_intensity(1.0*imread(image_path))
    im = adjust_sigmoid(equalize_adapthist(im))

    # store image shape before padding
    sh = im.shape
    im = pad_to_n(im, w=w)

    bw = np.zeros_like(im)
    imw = view_as_windows(im, (w,w), (w,w))
    imb = view_as_windows(bw, (w,w), (w,w))

    for i in range(imb.shape[0]):
        for j in range(imb.shape[1]):
            img = np.expand_dims(imw[i,j,...], axis=[0,3])
            # Todo: maybe move this somewhere else!
            p = await serve.get_deployment(model_deplyment_name).get_handle().predict.remote(img)
            p = p[0,:,:,0]
            b = p > 0.5
            imb[i,j,...] = b

    # revert back to original image shape
    im = im[:sh[0],:sh[1]]
    bw = bw[:sh[0],:sh[1]]
    bw = img_as_bool(bw)

    # postprocess
    bw = bw_watershed(bw)

    return bw, im

def imshow(im,title=''):
    h, w = im.shape[0:2]
    w, h = 1.6 * plt.figaspect(h/w)
    fig = plt.figure(figsize=(w,h))
    plt.imshow(im, interpolation=None, cmap='gray')
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    fig.canvas.manager.set_window_title(title)

def remove_big_objects(ar, min_size=64, connectivity=1):
    out = ar.copy()
    if min_size == 0:
        return out
    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out
    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are present.")
    too_big = component_sizes > min_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0
    return out

def plot_contours(bw, im, filename):
    bwrgb = label2rgb( label(bw, connectivity=1), image=im, alpha=0.1, image_alpha=1)
    contours = find_contours(bw, 0)
    imshow(bwrgb)
    ax = plt.gca()
    for c in contours:
        ax.plot(c[:,1],c[:,0],color='red',lw=1)
    plt.savefig(filename)
    plt.close('all')

def pad_to_n(im, w=64):
    # pad image with 0's in xy to a multiple of w
    if im.shape[0] % w == 0:
        r = 0
    else:
        r = w - (im.shape[0] % w)
    if im.shape[1] % w == 0:
        c = 0
    else:
        c = w - (im.shape[1] % w)
    return np.pad(im, ((0,r),(0,c)))

def bw_watershed(bw):
    bw = remove_small_holes(bw, 50)
    bw = remove_small_objects(bw, 20)
    distance = ndi.distance_transform_edt(bw, sampling=5)
    labels = label(bw, connectivity=1)
    coords = peak_local_max(distance, footprint=np.ones((5,5)), labels=labels, min_distance=5)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=labels, watershed_line=True)
    bw = remove_big_objects(bw, 250)
    bw = opening(labels > 0, disk(1))
    bw = remove_small_objects(bw, 25)
    return bw



def extract_regionprops(image_path, image_bw_path):

    props = ('area','area_bbox','axis_major_length','axis_minor_length','bbox',
             'centroid','eccentricity','extent','orientation','perimeter','solidity',
             'intensity_max', 'intensity_mean','intensity_min')
    im = imread(image_path)
    bw = label(imread(image_bw_path), connectivity=1)
    bwe = expand_labels(bw, distance=2)
    df_nuclei = pd.DataFrame(regionprops_table(bw, im, properties=props))
    df_cell = pd.DataFrame(regionprops_table(bwe, im, properties=props))
    df_membrane = pd.DataFrame(regionprops_table(bwe-bw, im, properties=props))
    df_nuclei = df_nuclei.add_prefix('nucleus_')
    df_cell = df_cell.add_prefix('cell_')
    df_membrane = df_membrane.add_prefix('membrane_')
    df = pd.concat([df_nuclei,df_cell,df_membrane], axis=1)
    return df


class unet:
    def __init__(self):
        self.model = load_model(os.path.join(root_dir, 'unet_default.h5'))
        self.logger = get_logger(root_dir + 'Unet')
        self.logger.info('model loaded')

    async def predict(self, img: np.ndarray):
        await serve.get_deployment("auto_scaler").get_handle().update_current_requests.remote(self.__class__.__name__)
        os.environ["OMP_NUM_THREADS"] = '1'
        p = self.model.predict(img)
        return p


async def predict_(class_name, task_id, resource_name, model_name, is_extract_regionprops,
                      window):
    await serve.get_deployment("auto_scaler").get_handle().update_current_requests.remote(class_name)

    data_dir = os.path.join(root_dir, 'dummy_data')
    image_path = os.path.join(data_dir, resource_name)
    image_ext = resource_name.split('.')[-1]
    output_dir = os.path.join('outputs', task_id)
    os.makedirs(output_dir, exist_ok=True)
    cache.hset(task_id, 'status', 'running eval')
    bw, im = await eval_images(image_path, "unet", w=window)

    bw_filepath = os.path.join(output_dir, resource_name.replace('.' + image_ext, '_bw.' + image_ext))
    contour_filpath = bw_filepath.replace('_bw', '_ov')
    imsave(bw_filepath, img_as_ubyte(bw), check_contrast=False)
    cache.hset(task_id, 'status', 'plotting contours')
    plot_contours(bw, im, contour_filpath)

    if is_extract_regionprops == 'True':
        cache.hset(task_id, 'status', 'extracting region properties')
        image_df = extract_regionprops(image_path, bw_filepath)
        image_df.to_csv(bw_filepath.replace('.' + image_ext, '.csv'))

    # zip the results
    cache.hset(task_id, 'status', 'zipping results')
    shutil.make_archive(output_dir, 'zip',
                        root_dir=output_dir)

@serve.ingress(app)
class alphan:
    def __init__(self):
        self.accepted_formats = ['png', 'tif', 'tiff']
        self.logger=get_logger(root_dir+'Alphan')
        self.logger.info('models loaded')

    @app.get("/predict", name="", summary="This endpoint predicts neuclei boundries in an image.")
    async def predict(self, background_tasks: BackgroundTasks, resource_name: str, model_name: str = 'unet_default', is_extract_regionprops: str = 'True',
                      window: int=128):

        task_id = str(uuid.uuid4()).replace('-', '')
        background_tasks.add_task(predict_, self.__class__.__name__, task_id, resource_name, model_name, is_extract_regionprops,
                      window)
        return {"get results at": "/results/get_results?task_id="+task_id,
                "check status at": "/results/task_status?task_id="+task_id}


    @app.get("/train", name="", summary="This endpoint trains a model to predict neuclei boundries in an image.")
    async def train(self):
        await serve.get_deployment("auto_scaler").get_handle().update_current_requests.remote(self.__class__.__name__)
        return None
