import base64
import pickle
from flask import Flask, request
app = Flask(__name__)

@app.route('/api/data')
def deserialize_data(request):
    '''
    get data from request. 
    Data passed is in base64 encode format.
    Decode the raw data and deserialize it to