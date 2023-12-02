from flask import Flask
from flask import render_template, redirect, url_for, request
import joblib
from datetime import datetime


app = Flask(__name__)
app.config['SECRET_KEY'] ='abc'

from Routes.routes import main
app.register_blueprint(main)
 
@app.route("/predecir", methods=["GET","POST"])
def PrediccionParams(): 
    """
        En esta funcion se carga el modelo y se recibe la info
        del JSON
    """
    if request.method == 'POST':
        x1 = request.form.get('x1', type=float) 
        x2 = request.form.get('x2', type=float)
        medidas = [x1, x2]
        clf = joblib.load('DataModel/model_rf.pkl') 
        predice = clf.predict([medidas])
        return 'La medida corresponde a la clase {0}'.format(predice[0])
    else: 
        return render_template("formulario.html")

if(__name__=="__main__"):
    app.run(debug = True) # corre la instancia de flask