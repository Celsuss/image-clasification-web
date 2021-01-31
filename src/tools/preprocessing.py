import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def get_preprocess_mode(modelname):
    # hardcode different preprocessing method for different models
    preprocessing_mode = {
    "resnet50": "caffe",
    "mobilenet": "tf",
    "xception": "tf"
    }

    for key in preprocessing_mode:
        if key in modelname:
            return preprocessing_mode[key]

def load_image(uploadpath, input_shape):
    """load image from disk and resize into the input_shape to corresponding model

    Args:
        uploadpath (str): path to where image is saved
        input_shape (tuple): shape to the model

    Returns:
        numpy.arrray: an image array
    """
    # 
    img = image.load_img(uploadpath, target_size=input_shape)

    img_arr = image.img_to_array(img)

    return img_arr


def preprocess_data(uploadpath, input_shape, modelname):

    img_arr = load_image(uploadpath, input_shape)

    return process(img_arr, modelname)


def process(image, modelname):
    """[summary]

    Args:
        image (np.array): iamge in array
        modelname (str): name of the model, e.x mobilenet, mobilenet_float16

    Returns:
        np.array: a preprocessed image array
    """

    x = np.expand_dims(image, axis=0)

    x = preprocess_input(x, mode=get_preprocess_mode(modelname))

    return x

