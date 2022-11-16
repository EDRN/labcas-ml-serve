import pandas as pd
from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.measure import label, find_contours, regionprops_table
from skimage.morphology import disk, remove_small_objects, opening, remove_small_holes
from skimage.segmentation import watershed, expand_labels
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread


def imshow(im, title=""):
    h, w = im.shape[0:2]
    w, h = 1.6 * plt.figaspect(h / w)
    fig = plt.figure(figsize=(w, h))
    plt.imshow(im, interpolation=None, cmap="gray")
    plt.axis("off")
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
    bwrgb = label2rgb(label(bw, connectivity=1), image=im, alpha=0.1, image_alpha=1)
    contours = find_contours(bw, 0)
    imshow(bwrgb)
    ax = plt.gca()
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color="red", lw=1)
    plt.savefig(filename)
    plt.close("all")


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
    return np.pad(im, ((0, r), (0, c)))


def bw_watershed(bw):
    bw = remove_small_holes(bw, 50)
    bw = remove_small_objects(bw, 20)
    distance = ndi.distance_transform_edt(bw, sampling=5)
    labels = label(bw, connectivity=1)
    coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=labels, min_distance=5)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=labels, watershed_line=True)
    bw = remove_big_objects(bw, 250)
    bw = opening(labels > 0, disk(1))
    bw = remove_small_objects(bw, 25)
    return bw


def extract_regionprops(image_path, image_bw_path):

    props = (
        "area",
        "area_bbox",
        "axis_major_length",
        "axis_minor_length",
        "bbox",
        "centroid",
        "eccentricity",
        "extent",
        "orientation",
        "perimeter",
        "solidity",
        "intensity_max",
        "intensity_mean",
        "intensity_min",
    )
    im = imread(image_path)
    bw = label(imread(image_bw_path), connectivity=1)
    bwe = expand_labels(bw, distance=2)
    df_nuclei = pd.DataFrame(regionprops_table(bw, im, properties=props))
    df_cell = pd.DataFrame(regionprops_table(bwe, im, properties=props))
    df_membrane = pd.DataFrame(regionprops_table(bwe - bw, im, properties=props))
    df_nuclei = df_nuclei.add_prefix("nucleus_")
    df_cell = df_cell.add_prefix("cell_")
    df_membrane = df_membrane.add_prefix("membrane_")
    df = pd.concat([df_nuclei, df_cell, df_membrane], axis=1)
    return df
