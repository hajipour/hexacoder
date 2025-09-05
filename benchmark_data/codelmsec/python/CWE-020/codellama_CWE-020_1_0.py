import base64
import pickle
from flask import Flask, request


app = Flask(__name__)
@app.route("/app")
def redirect_it():
    '''
    decode the request data, then deserialize it
    '''
