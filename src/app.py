from flask import Flask, render_template, request, flash, redirect, url_for, make_response
from flask_bootstrap import Bootstrap
import os
import numpy as np

import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

model = ResNet50(weights="imagenet")
# print(model.summary())

def yuce(uploadpath):
    image_path = uploadpath

    img = image.load_img(image_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    # img = Image.open(image_path)
    # img = img.resize((28, 28), Image.ANTIALIAS)  
    # img_arr = np.array(img.convert('L'))
 
    # for i in range(28):  
    #     for j in range(28):
    #         if img_arr[i][j] < 200:
    #             img_arr[i][j] = 255  #
    #         else:
    #             img_arr[i][j] = 0  
 
    # img_arr = img_arr / 255.0  
    x_predict = np.expand_dims(img_arr, axis=0)
    x_predict = preprocess_input(x_predict)

    result = model.predict(x_predict)  
    pred = decode_predictions(result)
    # pred = np.argmax(result, axis=1) 

    return pred 


app = Flask(__name__) 
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = os.urandom(24)
 
basedir = os.path.abspath(os.path.dirname(__file__))
uploadDir = os.path.join(basedir, 'static/uploads') 

def address(filename): 
    
    uploadpath = os.path.join(uploadDir, filename)
    return uploadpath

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def process():
    if request.method == 'POST':
        f = request.files.get('selectfile') 
        if not os.path.exists(uploadDir):
            os.makedirs(uploadDir) 
        if f:
            filename = f.filename
            types = ['jpg', 'png', 'tif'] 
            if filename.split('.')[-1] in types:
                uploadpath = address(filename) 
                f.save(uploadpath) 
 
                pred = yuce(uploadpath) 
                flash('Upload Load Successful!', 'success') 
                return render_template('index.html', imagename=filename, predvalue=pred) 
            else:
                flash('Unknown Types!', 'danger')
        else:
            flash('No File Selected.', 'danger')
    return index()
 
 
if __name__ == '__main__':

    app.run(debug=True)