import time
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from .preprocessing import preprocess_data, load_image

def classify_with_quantified(uploadpath, interpreter):

    start = time.time()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details['dtype'] == np.uint8:

        input_data = load_image(uploadpath, input_details["shape"][1:])
        input_scale, input_zero_point = input_details["quantization"]
        input_data = input_data / input_scale + input_zero_point
        input_data = np.expand_dims(input_data, axis=0).astype(input_details["dtype"])
    
    else:
        input_data = preprocess_data(uploadpath, input_details["shape"][1:])

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details["index"])
    end = time.time()

    pred = decode_predictions(result, top=3)

    return pred, end - start


def classify(uploadpath, model):

    start = time.time()
    x_predict = preprocess_data(uploadpath, model.input_shape[1:])

    result = model.predict(x_predict)  
    end = time.time()
    pred = decode_predictions(result, top=3)

    return pred, end - start
