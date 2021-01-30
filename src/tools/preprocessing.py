import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def get_preprocess_mode(modelname):

    preprocessing_mode = {
    "resnet50": "caffe",
    "mobilenet": "tf",
    "xception": "tf"
    }

    for key in preprocessing_mode:
        if key in modelname:
            return preprocessing_mode[key]

def load_image(uploadpath, input_shape):

    img = image.load_img(uploadpath, target_size=input_shape)

    img_arr = image.img_to_array(img)

    return img_arr


def preprocess_data(uploadpath, input_shape, modelname):

    img_arr = load_image(uploadpath, input_shape)

    return process(img_arr, modelname)

def process(image, modelname):

    x = np.expand_dims(image, axis=0)

    x = preprocess_input(x, mode=get_preprocess_mode(modelname))

    return x

