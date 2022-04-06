from keras.models import load_model
import os
load_model(os.path.join('models/alphan', 'unet_model_bg_gauss25.h5'))