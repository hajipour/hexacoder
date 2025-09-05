from flask import Flask
from flask import request
import logging


app = Flask(__name__)

@app.route("/")
def logger():
    '''
        Get the user input from the request and store it in a variable
        Log the user
