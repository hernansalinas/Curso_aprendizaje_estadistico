# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:22:08 2021

@author: david_hds
"""

from flask import Flask
from flask import render_template, redirect, url_for
from datetime import datetime


app = Flask (__name__)


from Routes.routes import *

if(__name__=="__main__"):
    app.run(debug = True)