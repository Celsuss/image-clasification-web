import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def preprocess_data(uploadpath, model):

    image_path = uploadpath

    # img = image.load_img(image_path, target_size=(224, 224))
    img = image.load_img(image_path, target_size=model.input_shape[1:])

    img_arr = image.img_to_array(img)

    x_predict = np.expand_dims(img_arr, axis=0)
    x_predict = preprocess_input(x_predict)

    return x_predict