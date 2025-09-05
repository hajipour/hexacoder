from flask import Flask
from flask import request
import logging


app = Flask(__name__)

@app.route("/")
def index():
    '''
    1. get the name from the request
    2. log the name as an info
