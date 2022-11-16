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

warnings.simplefilter(action="ignore", category=FutureWarning)

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__))))
from utils import extract_regionprops, bw_watershed, pad_to_n, plot_contours
from src.framework_utils import custom_docs

# Todo: store the redis ports and root paths in a config file!
cache = redis.Redis(host="localhost", port=6379, db=0)

app = FastAPI()

# edit the api docs, to add various info!
examples = [
    ("/predict", "post", 0, "Provide the name of the model to be used", ["unet_default"]),
    ("/predict", "post", 1, "Do you want to extract the region properties", ["True", "False"]),
    ("/predict", "post", 2, "Choose the image processing window size", [128, 64, 256]),
]
app.openapi = custom_docs(
    app,
    "Nuclei Position Detector by Alphan Altinok NASA, JPL, LabCAS ML Service (Beta)",
    "1.0.0",
    "Docs for Nuclei Position Detector by Alphan Altinok NASA, JPL",
    "/alphan",
    examples,
)

# todo: Fix the loggers paths and put it into common utils!
def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_path + ".log")
        logger.addHandler(fh)
    return logger


root_dir = "deployments/alphan/models"
data_dir = os.path.join(root_dir, "received_data")
os.makedirs(data_dir, exist_ok=True)


async def eval_images(image_path, model_deplyment_name="unet", w=64):

    im = rescale_intensity(1.0 * imread(image_path))
    im = adjust_sigmoid(equalize_adapthist(im))

    # store image shape before padding
    sh = im.shape
    im = pad_to_n(im, w=w)

    bw = np.zeros_like(im)
    imw = view_as_windows(im, (w, w), (w, w))
    imb = view_as_windows(bw, (w, w), (w, w))

    for i in range(imb.shape[0]):
        for j in range(imb.shape[1]):
            img = np.expand_dims(imw[i, j, ...], axis=[0, 3])
            # Todo: maybe move this somewhere else!
            p = await serve.get_deployment(model_deplyment_name).get_handle().predict.remote(img)
            p = p[0, :, :, 0]
            b = p > 0.5
            imb[i, j, ...] = b

    # revert back to original image shape
    im = im[: sh[0], : sh[1]]
    bw = bw[: sh[0], : sh[1]]
    bw = img_as_bool(bw)

    # postprocess
    bw = bw_watershed(bw)

    return bw, im


class unet:
    def __init__(self):
        self.model = load_model(os.path.join(root_dir, "unet_default.h5"))
        self.logger = get_logger(root_dir + "Unet")
        self.logger.info("model loaded")

    async def predict(self, img: np.ndarray):
        await serve.get_deployment("auto_scaler").get_handle().update_current_requests.remote(self.__class__.__name__)
        os.environ["OMP_NUM_THREADS"] = "1"
        p = self.model.predict(img)
        return p


async def predict_(class_name, task_id, resource_name, model_name, is_extract_regionprops, window):
    await serve.get_deployment("auto_scaler").get_handle().update_current_requests.remote(class_name)

    if model_name == "unet_default":
        model_deplyment_name = "unet"
    else:
        cache.hset(task_id, "status", "Error: the requested model has not been deployed yet!")
        return

    image_path = os.path.join(data_dir, resource_name)
    image_ext = resource_name.split(".")[-1]
    output_dir = os.path.join("outputs", task_id)
    os.makedirs(output_dir, exist_ok=True)
    cache.hset(task_id, "status", "running eval")
    bw, im = await eval_images(image_path, model_deplyment_name, w=window)

    bw_filepath = os.path.join(output_dir, resource_name.replace("." + image_ext, "_bw." + image_ext))
    contour_filpath = bw_filepath.replace("_bw", "_ov")
    imsave(bw_filepath, img_as_ubyte(bw), check_contrast=False)
    cache.hset(task_id, "status", "plotting contours")
    plot_contours(bw, im, contour_filpath)

    if is_extract_regionprops == "True":
        cache.hset(task_id, "status", "extracting region properties")
        image_df = extract_regionprops(image_path, bw_filepath)
        image_df.to_csv(bw_filepath.replace("." + image_ext, ".csv"))

    # zip the results
    cache.hset(task_id, "status", "zipping results")
    shutil.make_archive(output_dir, "zip", root_dir=output_dir)


@serve.ingress(app)
class alphan:
    def __init__(self):
        self.accepted_formats = ["png", "tif", "tiff"]
        self.logger = get_logger(root_dir + "Alphan")
        self.logger.info("models loaded")

    @app.post("/predict", name="", summary="This endpoint detects nuclei positions in an image.")
    async def predict(
        self,
        background_tasks: BackgroundTasks,
        input_image: UploadFile,
        model_name: str = "unet_default",
        is_extract_regionprops: str = "True",
        window: int = 128,
    ):

        # ref: https://stackoverflow.com/questions/63580229/how-to-save-uploadfile-in-fastapi
        async with aiofiles.open(os.path.join(data_dir, input_image.filename), "wb") as out_file:
            while content := await input_image.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        task_id = str(uuid.uuid4()).replace("-", "")
        background_tasks.add_task(
            predict_, self.__class__.__name__, task_id, input_image.filename, model_name, is_extract_regionprops, window
        )
        return {
            "get results at": "/results/get_results?task_id=" + task_id,
            "check status at": "/results/task_status?task_id=" + task_id,
        }

    @app.get("/train", name="", summary="This endpoint trains a model to detect nuclei positions in an image.")
    async def train(self):
        await serve.get_deployment("auto_scaler").get_handle().update_current_requests.remote(self.__class__.__name__)
        return None
