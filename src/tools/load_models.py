from .download_model import download
import os
import tensorflow as tf

def load():
    """function to load all models

    Returns:
        [dict]: {"$modelname": model_object}
    """
    models = {}

    res_net, mobile_net, x_ception = download(model_dict=models)

    PATH_TO_MODELS = "./processed_models/"

    for model in os.listdir(PATH_TO_MODELS):

        if model.endswith(".tflite"):
            interpreter = tf.lite.Interpreter(model_path=os.path.join(PATH_TO_MODELS, model))
            interpreter.allocate_tensors()

            models[model.split(".")[0]] = interpreter
        else:
            models[model.split(".")[0]] = tf.keras.models.load_model(os.path.join(PATH_TO_MODELS, model))
    
    return models