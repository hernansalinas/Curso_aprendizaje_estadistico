from flask import Flask
from flask import render_template, redirect, url_for
from datetime import datetime


app = Flask (__name__)
app.config['SECRET_KEY'] ='abc'


from routes import *

if(__name__=="__main__"):
    app.run(debug = True)