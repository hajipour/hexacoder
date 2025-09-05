from flask import Flask
from flask import request
import logging


app = Flask(__name__)

@app.route("/log")
def log():
     '''
     log a user name
     '''
