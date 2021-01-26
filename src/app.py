from flask import Flask, render_template, request, flash, redirect, url_for, make_response
from flask_bootstrap import Bootstrap
import os, time
import numpy as np
import tensorflow as tf

from tools.download_model import download
from tools.inference import classify, classify_with_quantified

res_net, mobile_net, x_ception = download()

model = res_net

supported_types = ['jpg', 'png', 'tif'] 

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
            if filename.split('.')[-1] in supported_types:
                uploadpath = address(filename) 
                f.save(uploadpath) 
 
                pred, t = classify(uploadpath, model) 
                flash('Upload Load Successful!', 'SUCESS') 
                return render_template('index.html', imagename=filename, predvalue=pred, used_time="{} seconds".format(t)) 
            else:
                flash('Unsupported File Types!', 'FAIL')
        else:
            flash('No File Selected.', 'FAIL')

    return index()
 
 
if __name__ == '__main__':

    app.run(debug=True)