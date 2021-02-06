import time
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from .preprocessing import preprocess_data, load_image

def classify(upload_path, modelname, model):
    """function to read an image and return prediction result given model

    Args:
        upload_path (str): path where uploaded image is saved
        modelname (str): name of the model
        model (model object): a model object to run inference on

    Returns:
        [tuple]: pred: the three output with highest likelihood
                 t: time used to return the result
    """
    start = time.time()

    if modelname.endswith("float16") or modelname.endswith("int8"):
        pred = classify_with_quantified(upload_path, modelname, model)
    
    else:
        pred = classify_original_model(upload_path, modelname, model)

    label, score = decode(decode_predictions(pred, top=5))

    end = time.time()
    
    return label, score, end - start


def decode(prediction):
    """decode prediction into label and score list

    Args:
        prediction ([type]): result from tensorflow decode_predictions

    Returns:
        [label]: a list of predictions, order by score
        [score]: a list of corresponding score, order by DESC
    """

    label = []
    score = []
    for res in prediction[0]:
        _, l, s = res
        label.append(l)
        score.append(str(s))
    
    return label, score

def classify_with_quantified(uploadpath, modelname, interpreter):
    """to run predicition on quantized models

    """
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
    """to run predicition on tf.keras.Model object

    """

    x_predict = preprocess_data(uploadpath, model.input_shape[1:], modelname)

    result = model.predict(x_predict)  

    return result
