import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from .preprocessing import preprocess_data

def classify_with_quantified(uploadpath, interpreter):

    x_predict = preprocess_data(uploadpath)

    input_data = x_predict.reshape(input_shape)
    interpreter.set_tensor(input_index, input_data)
    start = time.time()
    interpreter.invoke()
    end = time.time()
    result = interpreter.get_tensor(output_index)

    pred = decode_predictions(result)

    return pred, end - start


def classify(uploadpath, model):

    x_predict = preprocess_data(uploadpath, model)

    start = time.time()
    result = model.predict(x_predict)  
    end = time.time()
    pred = decode_predictions(result)

    return pred, end - start
