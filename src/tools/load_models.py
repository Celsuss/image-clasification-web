from .download_model import download
import os
import tensorflow as tf

def load():

    models = {}

    res_net, mobile_net, x_ception = download(model_dict=models)

    PATH_TO_LITE_MODELS = "./tflite_models/"

    for tflite_model in os.listdir(PATH_TO_LITE_MODELS):

        interpreter = tf.lite.Interpreter(model_path=os.path.join(PATH_TO_LITE_MODELS, tflite_model))
        interpreter.allocate_tensors()

        models[tflite_model.split(".")[0]] = interpreter
    
    return models