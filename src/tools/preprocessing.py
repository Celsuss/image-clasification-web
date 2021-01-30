import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

preprocessing_mode = {
    "resnet50": "caffe",
    "mobilenet": "tf",
    "xception": "tf"

}

def get_preprocess_mode(modelname):

    for key in preprocessing_mode:
        if key in modelname:
            return preprocessing_mode[key]

def load_image(uploadpath, input_shape):

    img = image.load_img(uploadpath, target_size=input_shape)

    img_arr = image.img_to_array(img)

    return img_arr


def preprocess_data(uploadpath, input_shape, modelname):

    img_arr = load_image(uploadpath, input_shape)

    x_predict = np.expand_dims(img_arr, axis=0)

    x_predict = preprocess_input(x_predict, mode=get_preprocess_mode(modelname))

    return x_predict

