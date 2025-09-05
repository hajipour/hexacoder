import base64
import pickle
from flask import Flask, request
app = Flask(__name__)

@app.route('/get_data')
def deserialize_object(request):
    '''
    get data from request, which is encoded by base64
    decode and de-serialize the data to obtained an object
