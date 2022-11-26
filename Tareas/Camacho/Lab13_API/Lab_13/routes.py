import json
from flask import render_template, jsonify
from flask import request, Response
from web_app import app
import joblib
# from sklearn.ensemble import RandomForestClassifier

@app.route("/")
def index():
   return render_template("index.html")

@app.route('/health', methods=['GET'])
def health():
   """
   This view is aimed to verify the healthyness of the API
   :return:
   """
   print('Hello')
   return Response('{"status":"OK"}', status=200)

def read_params_and_file_json(*args, **kwargs):
    a = kwargs["variable_entero"]
    b = kwargs["variable_string"]
    print(a,b)

    file_json = args[0]
    b = json.load(file_json["ArchivoJson"])
    print(b)
    return "done"

@app.route('/read_json', methods=['GET', 'POST'])
def read_json():
    read_params_and_file_json(request.files, **request.args)   
    return Response('{"status":"OK"}', status=200)


@app.route('/PrediccionParams', methods=['GET', 'POS'])
def prediccionParams():
   file = "DataModel/model_rf.pkl" 
   clf = joblib.load(file)
   