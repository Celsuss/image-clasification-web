import time
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from .preprocessing import preprocess_data, load_image

def classify(upload_path, modelname, model):

    start = time.time()

    if modelname.endswith("float16") or modelname.endswith("int8"):
        pred = classify_with_quantified(upload_path, modelname, model)
    
    else:
        pred = classify_original_model(upload_path, modelname, model)

    pred = decode_predictions(pred, top=3)

    end = time.time()

    return pred, end - start


def classify_with_quantified(uploadpath, modelname, interpreter):

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_data = preprocess_data(uploadpath, input_details["shape"][1:], modelname)

    if input_details['dtype'] == np.uint8:
        
        input_scale, input_zero_point = input_details["quantization"]
        input_data = input_data[0] / input_scale + input_zero_point
        input_data = np.expand_dims(input_data, axis=0).astype(input_details["dtype"])
    
    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details["index"])

    return result


def classify_original_model(uploadpath, modelname, model):

    x_predict = preprocess_data(uploadpath, model.input_shape[1:], modelname)

    result = model.predict(x_predict)  

    return result
