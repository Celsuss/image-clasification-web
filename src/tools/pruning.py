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

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

from tensorflow.keras import layers, optimizers, losses, callbacks
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile
from transfer import load_cifar
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.imagenet_utils import preprocess_input
# retrain_model = "resnet_cifar10.h5"

# model = tf.keras.models.load_model(retrain_model)

# x_train, y_train, x_validation, y_validation,x_test, y_test = load_cifar()

# def evaluate(model, x_train, y_train, x_validation, y_validation,x_test, y_test):
    
#     train = model.evaluate(x_train, y_train)
#     validation = model.evaluate(x_validation, y_validation)
#     test = model.evaluate(x_test, y_test)

#     return train, validation, test

# print(model.summary())
# print("baseline performance with {} parameters".format(model.count_params()))
# print(evaluate(model, x_train, y_train, x_validation, y_validation,x_test, y_test))
import tensorflow_datasets as tfds
# ds = tfds.load('imagenet2012_subset/10pct', split="validation")
from pathlib import Path

# ds = MyDataset()

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
        # for example in self.ds.batch(batch_size).take(1):   
              
            x, y = example["image"], example["label"]
            x = preprocess_input(tf.image.resize(example["image"], self.shape).numpy(), mode="tf")
            y = tf.keras.utils.to_categorical(example["label"], n_classes)
            yield x, y
            # yield x, tf.keras.utils.to_categorical(example["label"], n_classes)         


with mirrored_strategy.scope():
     
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

# print(model.evaluate(training_tf_generator))


training_tf_generator = tf.data.Dataset.from_generator(training_generator, output_shapes=output_shape, output_types=output_type)
# validation_tf_generator = tf.data.Dataset.from_generator(validation_generator.tf_generator, output_shapes=output_shape, output_types=output_type)

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

end_step = np.ceil(50000 / batch_size).astype(np.int32) * epochs
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer=optimizers.Adam(lr=2e-5), 
                loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
print(model_for_pruning.summary())
logdir = tempfile.mkdtemp()

callback = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

history = model_for_pruning.fit(training_tf_generator, epochs=epochs, batch_size=batch_size, 
                    callbacks=callback)


print("pruned performance with {} parameters".format(model_for_pruning.count_params()))
evaluate(model_for_pruning, x_train, y_train, x_validation, y_validation,x_test, y_test)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)