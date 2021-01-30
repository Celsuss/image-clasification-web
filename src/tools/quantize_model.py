import tensorflow as tf
from download_model import download
import pathlib
import tensorflow_datasets as tfds
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from preprocessing import process

ds = tfds.load('imagenet_v2', split="test")

class representative_data_gen():
    def __init__(self, shape, modelname):
        self.shape = shape
        self.modelname = modelname
    def generator(self):
        for example in ds.batch(1).take(128):
            x = process(tf.image.resize(example["image"], self.shape).numpy()[0], self.modelname)
            yield [x]

def quantize(model, quantize_level, path_to_save):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if quantize_level == tf.lite.OpsSet.TFLITE_BUILTINS_INT8:
        gen = representative_data_gen(model.input_shape[1:-1], model.name)
        converter.representative_dataset = gen.generator
        converter.target_spec.supported_ops = [quantize_level]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quantize_level == tf.float16:
        converter.target_spec.supported_types = [quantize_level]
    else:
        return 

    tflite_quant_model = converter.convert()
    path_to_save.write_bytes(tflite_quant_model)


if __name__ == "__main__":

    models = download()

    tflite_models_dir = pathlib.Path("./tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    quantize_level = {"int8": tf.lite.OpsSet.TFLITE_BUILTINS_INT8, 
                      #"float16": tf.float16
                      }

    for model in models:
        
        for q in quantize_level:

            quantize(model, quantize_level[q], tflite_models_dir/"{}_{}.tflite".format(model.name, q))




