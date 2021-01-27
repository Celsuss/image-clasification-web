import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input


def load_image(uploadpath, input_shape):

    img = image.load_img(uploadpath, target_size=input_shape)

    img_arr = image.img_to_array(img)

    return img_arr



def preprocess_data(uploadpath, input_shape):

    img_arr = load_image(uploadpath, input_shape)

    x_predict = np.expand_dims(img_arr, axis=0)
    x_predict = preprocess_input(x_predict)

    return x_predict