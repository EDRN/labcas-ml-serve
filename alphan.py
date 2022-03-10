import ray
from fastapi import FastAPI
from ray import serve
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.openapi.utils import get_openapi
from skimage.io import imread, imsave
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
from skimage.util import img_as_ubyte, img_as_float
from skimage.exposure import rescale_intensity
from skimage.morphology import remove_small_objects
import os
import numpy as np
from keras.models import load_model

app = FastAPI()

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
    # do changes:
    openapi_schema["paths"]["/predict"]['get']['parameters'][0]['examples']={
        'example1': {'value': '8_1.png'},
        'example2': {'value': '46_1.png'}
    }
    openapi_schema["paths"]["/predict"]['get']['parameters'][0]['description']="Give the LABCAS ID to the resource"

    openapi_schema["paths"]["/predict"]['get']['parameters'][1]['description'] = "Pre-processing step for background removal"
    openapi_schema["paths"]["/predict"]['get']['parameters'][1]['examples'] = {
        'example1': {'value': 'True'},
        'example2': {'value': 'False'}
    }
    openapi_schema["paths"]["/predict"]['get']['parameters'][2]['description'] = "Pre-processing step to include the rolling_ball algorithm"
    openapi_schema["paths"]["/predict"]['get']['parameters'][2]['examples'] = {
        'example1': {'value': 'True'},
        'example2': {'value': 'False'}
    }
    openapi_schema["paths"]["/predict"]['get']['parameters'][3]['description'] = "Chose the DNN model to do the prediction"
    openapi_schema["paths"]["/predict"]['get']['parameters'][3]['examples'] = {
        'example1': {'value': 'unet'},
        'example2': {'value': 'bgnet'}
    }
    openapi_schema["paths"]["/predict"]['get']['parameters'][4][
        'description'] = "Post-processing step to include Spur removal from the prediction"
    openapi_schema["paths"]["/predict"]['get']['parameters'][4]['examples'] = {
        'example1': {'value': 'remove_spur'},
        'example2': {'value': 'None'}
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

def remove_bg(x):
    bg = rolling_ball(x)
    return rescale_intensity(1.0 * (x - bg))

def remove_bg_gauss(x, sigma):
    imf = img_as_float(x)
    img = gaussian(imf, sigma=sigma)
    return rescale_intensity(1.0 * img_as_ubyte(imf - img))

def remove_spur(im):
    im = remove_small_objects(im, 64)
    return im

@serve.deployment
class Background_removal:
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray):
        img = remove_bg_gauss(img, sigma=15)
        return img

@serve.deployment
class Rolling_ball:
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray):
        img = remove_bg(img)
        return img

@serve.deployment
class Remove_spur:
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray):
        img = remove_bg(img)
        return img

@serve.deployment
class Bgnet:
    def __init__(self):
        self.root_dir = 'models/alphan'
        self.model = load_model(os.path.join(self.root_dir, 'unet_model_bg_gauss25.h5'))

    def __call__(self, img: np.ndarray):
        p = self.model.predict(img)
        # p = img
        return p

@serve.deployment
class Unet:
  def __init__(self):
      self.root_dir='models/alphan'
      self.model= load_model(os.path.join(self.root_dir, 'unet_model.h5'))

  def __call__(self, img: np.ndarray):
      p = self.model.predict(img)
      # p = img
      return p

@serve.deployment
@serve.ingress(app)
class Alphan:
    def __init__(self):
        self.count = 0
        self.accepted_formats = ['png', 'tif', 'tiff']
        self.root_dir = 'models/alphan'
        self.background_removal=Background_removal.get_handle()
        self.rolling_ball = Rolling_ball.get_handle()
        self.remove_spur = Remove_spur.get_handle()
        self.unet = Unet.get_handle()
        self.bgnet = Bgnet.get_handle()


    @app.get("/predict", name="", summary="This endpoint predicts neuclei boundries in an image.")
    async def predict(self, resource_name: str, background_removal: str = 'False', rolling_ball: str = 'False', model: str = 'unet',
            postprocess: str = 'remove_spur'):

        file_path = os.path.join(self.root_dir, 'dummy_data', resource_name)

        img = imread(file_path)

        if background_removal=='True':
            img = await self.background_removal.remote(img=img)

        if rolling_ball=='True':
            img = await self.rolling_ball.remote(img=img)

        img = rescale_intensity(1.0 * img)
        img = np.expand_dims(img, axis=[0, 3])

        if model == 'unet':
            p = await self.unet.remote(img)
        elif model == 'bgnet':
            p = await self.bgnet.remote(img)
        else:
            print('ERROR:', model, 'not available!')
            raise NotImplemented

        p = p[0, :, :, 0]
        b = p > 0.25

        if postprocess == 'remove_spur':
            b = await self.remove_spur.remote(b)

        imsave(file_path.replace('.png', '_out_bw.png'), img_as_ubyte(b))
        img = np.squeeze(img)
        alpha = 0.15
        zero = np.zeros_like(img)
        one = np.ones_like(img)
        img = np.stack((img, img, img, one), axis=2)
        mask = np.stack((zero, b * alpha, zero, 2 * b * alpha), axis=2)
        rgb = img + mask
        imsave(file_path.replace('.png', '_out_rgb.png'), rgb)
        return {"status:", "the result file was generated at: " + os.path.dirname(file_path)}

ray.init(address="auto", namespace="serve")
Background_removal.deploy()
Remove_spur.deploy()
Rolling_ball.deploy()
Unet.deploy()
Bgnet.deploy()
Alphan.deploy()

"""
STEPS to run:

1. Start local Ray cluster:
>> ray start --head 
2. Start Serve on the local Ray cluster.
>> serve start 
3. Run this script

ref: https://docs.ray.io/en/latest/serve/deployment.html
"""


"""
TODO: 
1. Create a new virtual env (maybe in a docker using this: https://stackoverflow.com/questions/24319662/from-inside-of-a-docker-container-how-do-i-connect-to-the-localhost-of-the-mach) and deploy some workers from there
2. Convert Alphan's code into new tensorflow 
    """