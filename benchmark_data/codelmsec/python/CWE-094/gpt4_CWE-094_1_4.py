import base64

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def saveName():
    '''
    Get first_name from the POST request, decode it using base64 and assign to variable.
