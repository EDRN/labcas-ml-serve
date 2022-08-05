import os
import sys
import glob2
import numpy as np
import pandas as pd

from scipy import ndimage as ndi

from skimage.color import rgb2gray, label2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist, adjust_sigmoid
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, threshold_isodata, threshold_sauvola
from skimage.io import imread, imsave
from skimage.measure import label, find_contours, regionprops_table
from skimage.morphology import disk, remove_small_objects, opening, remove_small_holes, binary_dilation
from skimage.segmentation import watershed, expand_labels
from skimage.util import img_as_float, img_as_ubyte, img_as_bool, view_as_windows

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merging import concatenate

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8,8)

import time
import uuid
import shutil

def imshow(im, title=''):
    h, w = im.shape[0:2]
    w, h = 1.6 * plt.figaspect(h / w)
    fig = plt.figure(figsize=(w, h))
    plt.imshow(im, interpolation=None, cmap='gray')
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    fig.canvas.manager.set_window_title(title)


def im_summarize(im):
    print('Shape:', im.shape)
    print('Type:', im.dtype)
    print('Min:', im.min())
    print('Max:', im.max())
    print('Size:', im.size)


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


def to_png(idir, odir):
    # TODO
    # placeholder for ensuring input image formats
    # accept additional image formats: TIFF's, PNG, ...
    # ensure the output is single channel grayscale images as PNG
    files = glob2.glob(os.path.join(idir, '*.tif'))
    for f in files:
        im = rgb2gray(imread(f))
        im = img_as_ubyte(rescale_intensity(im, out_range=(0, 1)))
        imsave(os.path.join(odir, os.path.basename(f).replace('.tif', '.png')), im, check_contrast=False)


def cp_annotations(idir, odir):
    # expects filename.png and filename_label.png pairs in idir
    files = glob2.glob(os.path.join(idir, '*[!~label].png'))
    for f in files:
        shutil.copy(os.path.join(idir, os.path.basename(f)), os.path.join(odir, os.path.basename(f)))
        l = f.replace('.png', '_label.png')
        shutil.copy(os.path.join(idir, os.path.basename(l)), os.path.join(odir, os.path.basename(l)))


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


################################################################
# multiple segmentation methods to generate training labels
# TODO include additional methods to augment training data ?
def thresh(im):
    im = img_as_float(im)
    im = rescale_intensity(im, out_range=(0, 1))
    gray = adjust_sigmoid(equalize_adapthist(im))
    bw = gray > 0.5
    bw = bw_watershed(bw)
    return bw


def local(im):
    return bw_watershed(im > threshold_local(im, block_size=65, method='gaussian'))


def isodata(im):
    return bw_watershed(im > threshold_isodata(im))


def sauvola(im):
    return bw_watershed(im > threshold_sauvola(im, window_size=75, k=0.1))


def plot_contours(bw, im, filename):
    bwrgb = label2rgb(label(bw, connectivity=1), image=im, alpha=0.1, image_alpha=1)
    contours = find_contours(bw, 0)
    imshow(bwrgb)
    ax = plt.gca()
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color='red', lw=1)
    plt.savefig(filename)
    plt.close('all')
    # imshow(bw)
    # ax = plt.gca()
    # for c in contours:
    #     ax.plot(c[:,1],c[:,0],color='magenta',lw=1)
    # plt.savefig(os.path.join(ovdir, os.path.basename(f).replace('.png','_thr_label.png')))
    # plt.close('all')


def make_labels(idir, odir, ovdir):
    files = glob2.glob(os.path.join(idir, '*.png'))
    for f in files:
        im = imread(f)

        fn_thr = os.path.join(odir, os.path.basename(f).replace('.png', '_thr.png'))
        fn_thr_mask = fn_thr.replace('.png', '_label.png')
        fn_thr_cont = os.path.join(ovdir, os.path.basename(f).replace('.png', '_thr.png'))

        fn_local = os.path.join(odir, os.path.basename(f).replace('.png', '_local.png'))
        fn_local_mask = fn_local.replace('.png', '_label.png')
        fn_local_cont = os.path.join(ovdir, os.path.basename(f).replace('.png', '_local.png'))

        fn_iso = os.path.join(odir, os.path.basename(f).replace('.png', '_iso.png'))
        fn_iso_mask = fn_iso.replace('.png', '_label.png')
        fn_iso_cont = os.path.join(ovdir, os.path.basename(f).replace('.png', '_iso.png'))

        fn_sau = os.path.join(odir, os.path.basename(f).replace('.png', '_sau.png'))
        fn_sau_mask = fn_sau.replace('.png', '_label.png')
        fn_sau_cont = os.path.join(ovdir, os.path.basename(f).replace('.png', '_sau.png'))

        bw = thresh(im)
        imsave(fn_thr, im, check_contrast=False)
        imsave(fn_thr_mask, img_as_ubyte(bw), check_contrast=False)
        plot_contours(bw, im, fn_thr_cont)

        bw = local(im)
        imsave(fn_local, im, check_contrast=False)
        imsave(fn_local_mask, img_as_ubyte(bw), check_contrast=False)
        plot_contours(bw, im, fn_local_cont)

        bw = isodata(im)
        imsave(fn_iso, im, check_contrast=False)
        imsave(fn_iso_mask, img_as_ubyte(bw), check_contrast=False)
        plot_contours(bw, im, fn_iso_cont)

        bw = sauvola(im)
        imsave(fn_sau, im, check_contrast=False)
        imsave(fn_sau_mask, img_as_ubyte(bw), check_contrast=False)
        plot_contours(bw, im, fn_sau_cont)


def make_tiles(idir, odir, w=64):
    files = glob2.glob(os.path.join(idir, '*[!~label].png'))
    for f in files:
        im = imread(f)
        mask = imread(f.replace('.png', '_label.png'))
        im = pad_to_n(im, w=w)
        bw = pad_to_n(mask, w=w)
        imw = view_as_windows(im, (w, w), (w, w))
        bww = view_as_windows(bw, (w, w), (w, w))
        for i in range(imw.shape[0]):
            for j in range(imw.shape[1]):
                filename = os.path.join(odir, os.path.basename(f).replace('.png', '_' + str(i) + '_' + str(j) + '.png'))
                maskname = filename.replace('.png', '_label.png')
                imsave(filename, imw[i, j, ...], check_contrast=False)
                imsave(maskname, bww[i, j, ...], check_contrast=False)

def read_images(folder):
    images = glob2.glob(os.path.join(folder,'*[!~label].png'))
    n_images = len(images)
    im = imread(images[0])
    [h,w] = im.shape
    x_images = np.zeros((n_images, h, w, 1), dtype=np.float64)
    y_images = np.zeros((n_images, h, w, 1), dtype=np.uint8)
    for n, f in enumerate(images):
        im = imread(f)
        lb = imread(f.replace('.png','_label.png'))
        x_images[n,:,:,0] = rescale_intensity(1.0*im)
        y_images[n,:,:,0] = lb/255.0
    return x_images, y_images


def make_unet(w=64):
    inputs = Input((w,w,1))

    s = Lambda(lambda x: x) (inputs)

    c1 = Conv2D(4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.1) (c3)
    c3 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.1) (c4)
    c4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.2) (c5)
    c5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    # check a 128x128 may help with hollow nuclei

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.1) (c6)
    c6 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.1) (c7)
    c7 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def eval_images(idir, odir, model, w=64):

    images = glob2.glob(os.path.join(idir,'*[!~label].png'))
    for f in images:
        im = rescale_intensity(1.0*imread(f))
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
                p = model.predict(img)
                p = p[0,:,:,0]
                b = p > 0.5
                # b = p > 0.25
                imb[i,j,...] = b

        # revert back to original image shape
        im = im[:sh[0],:sh[1]]
        bw = bw[:sh[0],:sh[1]]
        bw = img_as_bool(bw)

        # postprocess
        bw = bw_watershed(bw)

        filename = os.path.join(odir, os.path.basename(f).replace('.png','_bw.png'))
        contour_filename = filename.replace('_bw','_ov')
        imsave(filename, img_as_ubyte(bw), check_contrast=False)
        plot_contours(bw, im, contour_filename)

def train_model(idir, odir, modelname, w=64, split=0.9, batch=4, epoch=10):
    x_train, y_train = read_images(idir)

    print('X:', x_train.shape, x_train.min(), x_train.max(), x_train.dtype)
    print('Y:', y_train.shape, y_train.min(), y_train.max(), y_train.dtype)

    model = make_unet(w)

    # callbacks
    checkpoint = ModelCheckpoint(modelname,
                                monitor="val_loss",
                                mode="min",
                                save_best_only = True,
                                verbose=1)

    earlystop = EarlyStopping(monitor = 'val_loss',
                            min_delta = 0,
                            patience = 5,
                            verbose = 1,
                            restore_best_weights = True)

    results = model.fit(x_train, y_train,
                        validation_split=split,
                        batch_size=batch,
                        epochs=epoch,
                        callbacks=[earlystop, checkpoint])

    loss_values = results.history['loss']
    val_loss_values = results.history['val_loss']
    accuracy = results.history['accuracy']
    val_accuracy = results.history['val_accuracy']

    epochs = range(1, len(loss_values)+1)
    fig,ax = plt.subplots(1,2,figsize=(14,6))

    # accuracy v epochs
    ax[0].plot(epochs, accuracy, 'k', label='Training')
    ax[0].plot(epochs, val_accuracy, 'r', label='Validation')
    ax[0].set_title('Training & Validation Accuracy', fontsize=12)
    ax[0].set_xlabel('Epochs', fontsize=12)
    ax[0].set_ylabel('Accuracy', fontsize=12)
    ax[0].legend()

    # loss v epochs
    ax[1].plot(epochs, loss_values, 'k', label='Training')
    ax[1].plot(epochs, val_loss_values, 'r', label='Validation')
    ax[1].set_title('Training & Validation Loss', fontsize=12)
    ax[1].set_xlabel('Epochs', fontsize=12)
    ax[1].set_ylabel('Loss', fontsize=12)
    ax[1].legend()
    plt.savefig(modelname.replace('unet_model','learning_curves').replace('.h5','.png'))

    return model


def extract_regionprops(idir):

    props = ('area','area_bbox','axis_major_length','axis_minor_length','bbox',
             'centroid','eccentricity','extent','orientation','perimeter','solidity',
             'intensity_max', 'intensity_mean','intensity_min')

    images = glob2.glob(os.path.join(idir,'*[_bw].png'))
    for f in images:
        im = imread(f.replace('_bw',''))
        bw = label(imread(f), connectivity=1)
        bwe = expand_labels(bw, distance=2)
        df_nuclei = pd.DataFrame(regionprops_table(bw, im, properties=props))
        df_cell = pd.DataFrame(regionprops_table(bwe, im, properties=props))
        df_membrane = pd.DataFrame(regionprops_table(bwe-bw, im, properties=props))
        df_nuclei = df_nuclei.add_prefix('nucleus_')
        df_cell = df_cell.add_prefix('cell_')
        df_membrane = df_membrane.add_prefix('membrane_')
        df = pd.concat([df_nuclei,df_cell,df_membrane], axis=1)
        df.to_csv(f.replace('.png', '.csv'))