import tensorflow as tf
from download_model import download
import pathlib
import tensorflow_datasets as tfds
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from preprocessing import process

ds = tfds.load('imagenet2012_subset/1pct', split="train")

class representative_data_gen():
    # to generate TF dataset for quantizing int8 models
    def __init__(self, shape, modelname):
        self.shape = shape
        self.modelname = modelname
    def generator(self):
        for example in ds.batch(1).take(128):
            x = process(tf.image.resize(example["image"], self.shape).numpy()[0], self.modelname)
            yield [x]

def quantize(model, quantize_level, path_to_save):
    """function to quantize a keras model 

    Args:
        model (tf.keras.Model): a keras application model
        quantize_level (tf. data type): choose from [ tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.float16]
        path_to_save (str): path to save converted model
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if quantize_level == tf.lite.OpsSet.TFLITE_BUILTINS_INT8:
        gen = representative_data_gen(model.input_shape[1:-1], model.name)
        converter.representative_dataset = gen.generator
        # need to specify int type when quantize into int models
        converter.target_spec.supported_ops = [quantize_level]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quantize_level == tf.float16:
        converter.target_spec.supported_types = [quantize_level]
    else:
        return 
    # concert model and save to disk
    tflite_quant_model = converter.convert()
    path_to_save.write_bytes(tflite_quant_model)


if __name__ == "__main__":

    # return all models to be quantized 
    models = download()

    tflite_models_dir = pathlib.Path("./tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    quantize_level = {"int8": tf.lite.OpsSet.TFLITE_BUILTINS_INT8, 
                      "float16": tf.float16
                      }

    for model in models:
        
        for q in quantize_level:

            quantize(model, quantize_level[q], tflite_models_dir/"{}_{}.tflite".format(model.name, q))




