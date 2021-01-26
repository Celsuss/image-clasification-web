import tensorflow as tf

def quantize(model):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    tflite_quant_model = converter.convert()
    open("converted_quant_model.tflite", "wb").write(tflite_quant_model)
    interpreter = tf.lite.Interpreter(model_path="converted_quant_model.tflite")
    interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
input_shape = input_details[0]['shape']

