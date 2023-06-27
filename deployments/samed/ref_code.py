from importlib import import_module
from segment_anything_sv import sam_model_registry as sam_model_registry_local
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torch
from scipy.ndimage import zoom
from einops import repeat
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import supervision as sv
import urllib.request

class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach',
                 7: 'aorta', 8: 'pancreas'}

def init():
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "sam_vit_b_01ec64.pth")
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                               "sam_vit_h_4b8939.pth")

def run_samed():
    # preprocessing of images? : https://nipy.org/nibabel/
    # ref: https://github.com/hitachinsk/SAMed/tree/main
    ckpt = 'checkpoints/sam_vit_b_01ec64.pth'
    lora_ckpt = 'checkpoints/epoch_159.pth'
    vit_name = 'vit_b'
    img_size = 512
    num_classes = 8
    rank = 4
    multimask_output = True
    patch_size = [img_size, img_size]
    sam, img_embedding_size = sam_model_registry_local[vit_name](image_size=img_size,
                                                           num_classes=num_classes,
                                                           checkpoint=ckpt, pixel_mean=[0, 0, 0],
                                                           pixel_std=[1, 1, 1])
    pkg = import_module('sam_lora_image_encoder')
    model = pkg.LoRA_Sam(sam, rank)
    model.load_lora_parameters(lora_ckpt)
    image=np.load('data/77slice.npy')
    x, y = image.shape[0], image.shape[1]
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
    inputs = torch.from_numpy(image).unsqueeze(
        0).unsqueeze(0).float()
    inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
    with torch.no_grad():
        outputs = model(inputs, multimask_output, patch_size[0])
        output_masks = outputs['masks']
        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        pred = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            pred = zoom(pred, (x / patch_size[0], y / patch_size[1]), order=0)
    plt.figure()
    plt.imshow(pred)
    plt.savefig('pred_samed.png')
    return pred

def run_sam():
    # ref: pip install 'git+https://github.com/facebookresearch/segment-anything.git'
    # model at: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # get https://media.roboflow.com/notebooks/examples/dog.jpeg
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    IMAGE_PATH='data/77slice.png'
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blank_image=np.full(image.shape, 255, dtype=int)
    sam_result = mask_generator.generate(image)
    mask_annotator = sv.MaskAnnotator()
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotations = mask_annotator.annotate(scene=blank_image, detections=detections)
    plt.figure()
    plt.imshow(annotations)
    plt.savefig('pred_sam.png')
    return annotated_image

if __name__ == '__main__':
   run_samed()
   run_sam()