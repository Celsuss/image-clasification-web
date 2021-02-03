import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers, optimizers, losses, callbacks
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

res_net = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

def load_cifar():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    return x_train, y_train, x_validation, y_validation,x_test, y_test

def build_model():

    model = Sequential()
    model.add(layers.UpSampling2D((2,2)), input_shape=(224,224,3))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(lr=2e-5), 
                loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == "__main__":

    output_path = "resnet_cifar10.h5"

    x_train, y_train, x_validation, y_validation,x_test, y_test = load_cifar()

    model = build_model()

    callback = [callbacks.ModelCheckpoint(output_path, verbose=1, save_best_only=True),
                callbacks.EarlyStopping(patience=5)]

    history = model.fit(x_train, y_train, epochs=15, batch_size=128, 
                    callbacks=callback, validation_data=(x_validation, y_validation))
