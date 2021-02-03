import tensorflow as tf

##################################
# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of 10% the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)
###################################

from tensorflow.keras import layers, optimizers, losses, callbacks
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile
from transfer import load_cifar
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds

ds_tr = tfds.load('imagenet_resized/64x64', split="train")
ds_val = tfds.load('imagenet_resized/64x64', split="validation")

batch_size = 64
epochs = 10
n_classes = 1000

output_path = "retrain_mobilenet.h5"

class representative_data_gen():
    # to generate TF dataset for quantizing int8 models
    def __init__(self, shape, modelname, ds, size):
        self.shape = shape
        self.modelname = modelname
        self.ds = ds
        self.size = size
    def generator(self):

        for example in self.ds.batch(batch_size).take(self.size // batch_size):   
              
            x, y = example["image"], example["label"]
            x = preprocess_input(tf.image.resize(example["image"], self.shape).numpy(), mode="tf")
            y = tf.keras.utils.to_categorical(example["label"], n_classes)
            yield x, y

# use both gpu to speed up    
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

with mirrored_strategy.scope():

    # this file is used to retrain mobilenet on a resized imagenet dataset (because this dataset is smaller and no need for manual download)
     
    model = MobileNet(weights='imagenet')

    output_shape = ((batch_size, *model.input_shape[1:]), (batch_size, n_classes))
    output_type = (tf.float32, tf.int32)

    gen_tr = representative_data_gen(model.input_shape[1:-1], "mobilenet", ds_tr, size=600000)
    training_generator = gen_tr.generator
    training_tf_generator = tf.data.Dataset.from_generator(training_generator, output_shapes=output_shape, output_types=output_type)

    gen_val = representative_data_gen(model.input_shape[1:-1], "mobilenet", ds_val, size=50000)
    val_generator = gen_val.generator
    val_tf_generator = tf.data.Dataset.from_generator(val_generator, output_shapes=output_shape, output_types=output_type)

    model.compile(optimizer=optimizers.Adam(lr=0.001), 
                loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [callbacks.ModelCheckpoint(output_path, verbose=1, save_best_only=True),
                callbacks.EarlyStopping(patience=5)]

    history = model.fit(training_tf_generator, epochs=epochs, batch_size=batch_size, 
                    callbacks=callback, validation_data=val_tf_generator)

