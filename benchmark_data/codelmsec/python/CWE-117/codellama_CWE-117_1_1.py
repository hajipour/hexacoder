from flask import Flask
from flask import request
import logging


app = Flask(__name__)

logging.basicConfig(filename='flask.log', level=logging.INFO)
@app.before_request
def before_request():
   
