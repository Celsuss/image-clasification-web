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
import os, time
import tensorflow_model_optimization as tfmot
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds

# ds_tr = tfds.load('imagenet2012_subset/1pct', split="train").repeat().prefetch(1)
# ds_val = tfds.load('imagenet2012_subset/1pct', split="validation").repeat().prefetch(1)

ds_tr = tfds.load('imagenet2012', split="train").repeat().prefetch(1)
ds_val = tfds.load('imagenet2012', split="validation").repeat().prefetch(1)

batch_size = 100
epochs = 20
n_classes = 1000
train_size = 800000
validation_size = 50000

class data_gen():
    # to generate TF dataset for quantizing int8 models
    def __init__(self, shape, modelname, ds, size):
        self.shape = shape
        self.modelname = modelname
        self.ds = ds
        self.size = size

    def generator(self):
        x = np.zeros((batch_size, *self.shape, 3))
        y = np.zeros((batch_size, n_classes))
        while True:
            count = 0 
            for example in self.ds:
                x[count % batch_size] = preprocess_input(tf.image.resize(example["image"], self.shape).numpy(), mode="tf")
                y[count % batch_size] = np.expand_dims(tf.keras.utils.to_categorical(example["label"], num_classes=n_classes), 0)
                count += 1
                if (count % batch_size == 0):
                    yield x, y


with mirrored_strategy.scope():
     
    model = MobileNet(weights='imagenet')

    output_shape = ((batch_size, *model.input_shape[1:]), (batch_size, n_classes))
    output_type = (tf.float32, tf.int32)

    gen_tr = data_gen(model.input_shape[1:-1], "mobilenet", ds_tr, size=train_size)
    training_generator = gen_tr.generator
    training_tf_generator = tf.data.Dataset.from_generator(training_generator, output_shapes=output_shape, output_types=output_type).take(train_size // batch_size)

    gen_val = data_gen(model.input_shape[1:-1], "mobilenet", ds_val, size=validation_size)
    val_generator = gen_val.generator
    val_tf_generator = tf.data.Dataset.from_generator(val_generator, output_shapes=output_shape, output_types=output_type).take(validation_size // batch_size)

    model.compile(optimizer=optimizers.Adam(lr=0.001), 
                loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    train_acc = model.evaluate(training_tf_generator, workers=64,
               use_multiprocessing=True)
    end_time1 = time.time()
    val_acc = model.evaluate(val_tf_generator, workers=64,
               use_multiprocessing=True)
    end_time2 = time.time()
    with open("result.log", "w") as f:

        f.write("original model performance, train acc {}, use time {}, val acc {}, use_time {}\n".format(
                    train_acc[1], end_time1-start_time, val_acc[1], end_time2-end_time1))

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    end_step = np.ceil(train_size / batch_size).astype(np.int32) * epochs
    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                               final_sparsity=0.70,
                                                               begin_step=0,
                                                               end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer=optimizers.Adam(lr=2e-5), 
                loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    print(model_for_pruning.summary())
    logdir = "./logs"
    os.makedirs(logdir, exist_ok=True)

    output_path = os.path.join(logdir, "model_checkpoint.h5")
    callback = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    callbacks.ModelCheckpoint(output_path, verbose=1, save_best_only=True),
    callbacks.EarlyStopping(patience=5), callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]

    history = model_for_pruning.fit(training_tf_generator, epochs=epochs, batch_size=batch_size, 
                    callbacks=callback, validation_data=val_tf_generator, workers=64,
                    use_multiprocessing=True)

    print("pruned performance with {} parameters".format(model_for_pruning.count_params()))

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    pruned_keras_file = "pruned_mobilenet.h5"
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)

    model_pruned = tf.keras.models.load_model(pruned_keras_file)
    model_pruned.compile(optimizer=optimizers.Adam(lr=2e-5), 
                loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    train_acc_prune = model_pruned.evaluate(training_tf_generator, workers=64,
               use_multiprocessing=True)
    end_time1 = time.time()
    val_acc_prune = model_pruned.evaluate(val_tf_generator, workers=64,
               use_multiprocessing=True)
    end_time2 = time.time()

    with open("result.log", "a") as f:

        f.write("pruned model performance, train acc {}, use time {}, val acc {}, use_time {}\n".format(
                    train_acc_prune[1], end_time1-start_time, val_acc_prune[1], end_time2-end_time1))
