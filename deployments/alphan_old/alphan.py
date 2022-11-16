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
import logging
from fastapi import FastAPI


def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_path + ".log")
        logger.addHandler(fh)
    return logger


root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

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

    # START: ====== predict endpoint info

    # param 1 info
    openapi_schema["paths"]["/predict"]["get"]["parameters"][0]["description"] = "Give the LABCAS ID to the resource"
    openapi_schema["paths"]["/predict"]["get"]["parameters"][0]["examples"] = {
        "example1": {"value": "8_1.png"},
        "example2": {"value": "46_1.png"},
    }

    # param 2 info
    openapi_schema["paths"]["/predict"]["get"]["parameters"][1][
        "description"
    ] = "Pre-processing step for background removal"
    openapi_schema["paths"]["/predict"]["get"]["parameters"][1]["examples"] = {
        "example1": {"value": "True"},
        "example2": {"value": "False"},
    }

    # param 3 info
    openapi_schema["paths"]["/predict"]["get"]["parameters"][2][
        "description"
    ] = "Pre-processing step to include the rolling_ball algorithm"
    openapi_schema["paths"]["/predict"]["get"]["parameters"][2]["examples"] = {
        "example1": {"value": "True"},
        "example2": {"value": "False"},
    }

    # param 4 info
    openapi_schema["paths"]["/predict"]["get"]["parameters"][3][
        "description"
    ] = "Chose the DNN model to do the prediction"
    openapi_schema["paths"]["/predict"]["get"]["parameters"][3]["examples"] = {
        "example1": {"value": "unet"},
        "example2": {"value": "bgnet"},
    }

    # param 5 info
    openapi_schema["paths"]["/predict"]["get"]["parameters"][4][
        "description"
    ] = "Post-processing step to include Spur removal from the prediction"
    openapi_schema["paths"]["/predict"]["get"]["parameters"][4]["examples"] = {
        "example1": {"value": "remove_spur"},
        "example2": {"value": "None"},
    }

    # END: ====== predict endpoint info

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def remove_spur(im):
    im = remove_small_objects(im, 64)
    return im


def remove_bg(x):
    bg = rolling_ball(x)
    bg_removed = rescale_intensity(1.0 * (x - bg))
    return bg_removed


def remove_bg_gauss(x, sigma):
    imf = img_as_float(x)
    img = gaussian(imf, sigma=sigma)
    removed = rescale_intensity(1.0 * img_as_ubyte(imf - img))
    return removed


class bgnet:
    def __init__(self):
        self.model = load_model(os.path.join(root_dir, "unet_model_bg_gauss25.h5"))
        self.logger = get_logger(root_dir + "Bgnet")
        self.logger.info("model loaded")

    async def predict(self, img: np.ndarray):
        await serve.get_deployment("auto_scaler").get_handle().update_current_requests.remote(self.__class__.__name__)
        os.environ["OMP_NUM_THREADS"] = "1"  # https://docs.ray.io/en/latest/serve/core-apis.html#serve-cpus-gpus
        p = self.model.predict(img)
        return p


class unet:
    def __init__(self):
        self.model = load_model(os.path.join(root_dir, "unet_model.h5"))
        self.logger = get_logger(root_dir + "Unet")
        self.logger.info("model loaded")

    async def predict(self, img: np.ndarray):
        await serve.get_deployment("auto_scaler").get_handle().update_current_requests.remote(self.__class__.__name__)
        os.environ["OMP_NUM_THREADS"] = "1"
        p = self.model.predict(img)
        return p


@serve.ingress(app)
class alphan:
    def __init__(self):
        self.accepted_formats = ["png", "tif", "tiff"]
        self.logger = get_logger(root_dir + "Alphan")
        # self.unet_model=ray.put(load_model(os.path.join(root_dir, 'unet_model.h5')))
        # self.bgnet_model = ray.put(load_model(os.path.join(root_dir, 'unet_model_bg_gauss25.h5')))
        self.logger.info("models loaded")

    @app.get("/predict", name="", summary="This endpoint predicts neuclei boundries in an image.")
    async def predict(
        self,
        resource_name: str,
        remove_background: str = "False",
        apply_rolling_ball: str = "False",
        model_name: str = "unet",
        postprocess: str = "remove_spur",
    ):

        await serve.get_deployment("auto_scaler").get_handle().update_current_requests.remote(self.__class__.__name__)

        file_path = os.path.join(root_dir, "dummy_data", resource_name)

        img = imread(file_path)

        if remove_background == "True":
            img = await serve.get_deployment("remove_bg_gauss").get_handle().remote(img=img)

        if apply_rolling_ball == "True":
            img = await serve.get_deployment("remove_bg").get_handle().remote(img=img)

        img = rescale_intensity(1.0 * img)
        img = np.expand_dims(img, axis=[0, 3])

        if model_name.lower() == "unet":
            p = await serve.get_deployment("unet").get_handle().predict.remote(img)

        elif model_name == "bgnet":
            p = await serve.get_deployment("bgnet").get_handle().predict.remote(img)
        else:
            print("ERROR:", model_name, "not available!")
            raise NotImplemented

        p = p[0, :, :, 0]
        b = p > 0.25

        if postprocess == "remove_spur":
            b = await serve.get_deployment("remove_spur").get_handle().remote(b)

        imsave(file_path.replace(".png", "_out_bw.png"), img_as_ubyte(b))
        img = np.squeeze(img)
        alpha = 0.15
        zero = np.zeros_like(img)
        one = np.ones_like(img)
        img = np.stack((img, img, img, one), axis=2)
        mask = np.stack((zero, b * alpha, zero, 2 * b * alpha), axis=2)
        rgb = img + mask
        imsave(file_path.replace(".png", "_out_rgb.png"), rgb)
        return {"status:", "the result file was generated at: " + os.path.dirname(file_path)}
