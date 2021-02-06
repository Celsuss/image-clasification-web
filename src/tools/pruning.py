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
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds

# ds_tr = tfds.load('imagenet2012_subset/1pct', split="train").repeat().prefetch(1)
# ds_val = tfds.load('imagenet2012_subset/1pct', split="validation").repeat().prefetch(1)

ds_tr = tfds.load('imagenet2012', split="train")
ds_val = tfds.load('imagenet2012', split="validation")

batch_size = 64
epochs = 10
n_classes = 1000
train_size = 600000
validation_size = 50000

class representative_data_gen():
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
            for example in self.ds.shuffle(self.size):
                x[count % batch_size] = preprocess_input(tf.image.resize(example["image"], self.shape).numpy(), mode="tf")
                y[count % batch_size] = np.expand_dims(tf.keras.utils.to_categorical(example["label"], num_classes=n_classes), 0)
                count += 1
                if (count % batch_size == 0):
                    yield x, y
        # while True:
        #     for example in self.ds.shuffle(self.size):
        #         x, y = example["image"], example["label"]
        #         x = preprocess_input(tf.image.resize(example["image"], self.shape).numpy(), mode="tf")
        #         y = tf.keras.utils.to_categorical(example["label"], n_classes)
        #         yield x, y


with mirrored_strategy.scope():
     
    model = MobileNet(weights='imagenet')

    output_shape = ((batch_size, *model.input_shape[1:]), (batch_size, n_classes))
    output_type = (tf.float32, tf.int32)

    gen_tr = representative_data_gen(model.input_shape[1:-1], "mobilenet", ds_tr, size=train_size)
    training_generator = gen_tr.generator
    training_tf_generator = tf.data.Dataset.from_generator(training_generator, output_shapes=output_shape, output_types=output_type).take(train_size // batch_size)

    gen_val = representative_data_gen(model.input_shape[1:-1], "mobilenet", ds_val, size=validation_size)
    val_generator = gen_val.generator
    val_tf_generator = tf.data.Dataset.from_generator(val_generator, output_shapes=output_shape, output_types=output_type).take(validation_size // batch_size)

    model.compile(optimizer=optimizers.Adam(lr=0.001), 
                loss='categorical_crossentropy', metrics=['accuracy'])

    train_acc = model.evaluate(training_tf_generator, workers=64,
               use_multiprocessing=True)
    val_acc = model.evaluate(val_tf_generator, workers=64,
               use_multiprocessing=True)

    with open("result.log", "w") as f:

        f.write("original model performance with parameter, train acc {}, val acc {}\n".format(model.count_params(), train_acc[1], val_acc[1]))

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    end_step = np.ceil(600000 / batch_size).astype(np.int32) * epochs
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
                    callbacks=callback, validation_data=val_tf_generator, workers=64,
                    use_multiprocessing=True)

    print("pruned performance with {} parameters".format(model_for_pruning.count_params()))

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)

    train_acc_prune = model.evaluate(training_tf_generator, workers=64,
               use_multiprocessing=True)
    eval_acc_prune = model.evaluate(val_tf_generator, workers=64,
               use_multiprocessing=True)

    with open("result.log", "a") as f:

        f.write("original model performance with parameter, train acc {}, val acc {}\n".format(model_for_pruning.count_params(), train_acc_prune[1], eval_acc_prune[1]))

