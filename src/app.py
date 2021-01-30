from flask import Flask, request, make_response, jsonify
from werkzeug.utils import secure_filename
import os, time
from tools.inference import classify
from flask_cors import CORS
from tools.load_models import load

models = load()

supported_types = ['jpg', 'png'] 

app = Flask(__name__) 

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.config['SECRET_KEY'] = os.urandom(24)
 
basedir = os.path.abspath(os.path.dirname(__file__))
uploadDir = os.path.join(basedir, 'static/uploads') 

def address(filename): 
    
    uploadpath = os.path.join(uploadDir, filename)
    return uploadpath

@app.route('/')
def index():
    return 'Welcome!'

@app.route('/test', methods=['GET'])
def test():
    return 'Just a test!'

@app.route('/list_model', methods=['GET'])
def list_model():

    supported_models = []
    for m in models:
        supported_models.append(m)
    
    return make_response(jsonify({"models": supported_models}), 200)

@app.route('/post_image', methods=['POST'])
def testPost():
    f = request.files['file']
    if not os.path.exists(uploadDir):
            os.makedirs(uploadDir) 
    if f:
        filename = f.filename
        if filename.split('.')[-1] in supported_types:

            filename = secure_filename(filename)
            uploadpath = address(filename) 
            f.save(uploadpath) 

            chosen_model = request.form['model_name']

            pred, t = classify(uploadpath, chosen_model, models[chosen_model]) 
            
            print(pred, t)
            _, prediction, probability = pred[0][0]

            res = make_response(jsonify({"status": "SUCCESS", "prediction": str(prediction), 
                                    "likelihood": str(probability), "used_time": str(t)}), 200)
        else:
            res = make_response(jsonify({"status": "FAIL", "msg": "Unspported image file format"}), 406)
    else:
        res = make_response(jsonify({"status": "FAIL", "msg": "No file uploaded"}), 400)

    return res

 
if __name__ == '__main__':

    app.run(debug=True)