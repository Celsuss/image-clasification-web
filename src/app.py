from flask import Flask, render_template, request, flash, redirect, url_for, make_response, jsonify
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
    return {'Value' : 'Welcome'}

@app.route('/test', methods=['GET'])
def test():
    return {'Value' : 'Hello World'}

@app.route('/list_model', methods=['GET'])
def list_model():

    supported_models = []
    for m in models:
        supported_models.append(m)
    
    return make_response(jsonify({"models": supported_models}), 200)

    
@app.route('/testPost', methods=['POST'])
def testPost():
    f = request.files['file']
    if not os.path.exists(uploadDir):
            os.makedirs(uploadDir) 
    if f:
        filename = f.filename
        if filename.split('.')[-1] in supported_types:
            uploadpath = address(filename) 
            f.save(uploadpath) 

            #TODO: get the which model to run prediction
            # chosen_model = request.body["model"]
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

# @app.route('/', methods=['POST', 'GET'])
# def process():
#     if request.method == 'POST':
#         f = request.files.get('selectfile') 
#         if not os.path.exists(uploadDir):
#             os.makedirs(uploadDir) 
#         if f:
#             filename = f.filename
#             if filename.split('.')[-1] in supported_types:
#                 uploadpath = address(filename) 
#                 f.save(uploadpath) 
 
#                 pred, t = classify(uploadpath, model)  # classify_with_quantified(uploadpath, interpreter) # classify(uploadpath, model) 
#                 flash('Upload Load Successful!', 'SUCESS') 
#                 return render_template('index.html', imagename=filename, predvalue=pred, used_time="{} seconds".format(t)) 
#             else:
#                 flash('Unsupported File Types!', 'FAIL')
#         else:
#             flash('No File Selected.', 'FAIL')

#     return index()
 
 
if __name__ == '__main__':

    app.run(debug=True)