
import json
from Web_app import app
from flask import render_template,jsonify
#from MainExecution import *
from flask import request, Response
#import forms
import numpy as np
import pickle
import joblib


#-------------------------------------------------------------
@app.route("/")
def index():
   return render_template("index.html")

   
#-------------------------------------------------------------   
@app.route('/health', methods=['GET'])
def health():
   """
   This view is aimed to verify the healthyness of the API
   :return:
   """
   return Response('{"status":"OK"}', status=200)
   

 #-------------------------------------------------------------
@app.route("/PrediccionParams", methods=['GET', 'POST'])

def requestCoord():
   pred = PredictParams(**request.args)
   return render_template("predecir.html", value = pred) 


def PredictParams(**kwargs):

   model_rf_joblib = joblib.load('/media/alejandro/d473ee28-d240-47f3-b97d-203933cc0842/home/usuario/Downloads/Statistical Learning/Modelo/DataModel/model_rf.pkl') 

   a = kwargs["X1"]
   b = kwargs["X2"]

   return model_rf_joblib.predict(np.array([[a,b]]))[0]

#-------------------------------------------------------------

@app.route('/read_json', methods=['GET', 'POST'])
def read_json():

   read_params_and_file_json(request.files, **request.args)  #request.files, 
   return Response('{"status":"OK"}', status=200)

def read_params_and_file_json(*args, **kwargs):   

   a = kwargs["variable_entero"]
   b = kwargs["variable_string"]
   print(a,b)

   file_json = args[0]
   b = json.load(file_json["ArchivoJson"])
   print(b)

   return "done"

#-------------------------------------------------------------

@app.route("/PrediccionJson", methods=['GET', 'POST'])

def requestArchive():
   pred = PredictParamsJson(request.files)
   return render_template("predecir.html", value = pred) 

def PredictParamsJson(*args):

   file_json = args[0]
   b = json.load(file_json["ArchivoJson"])

   model_rf_joblib = joblib.load('/media/alejandro/d473ee28-d240-47f3-b97d-203933cc0842/home/usuario/Downloads/Statistical Learning/Modelo/DataModel/model_rf.pkl') 

   return model_rf_joblib.predict(np.array([[b["X1"],b["X2"]]]))[0]




