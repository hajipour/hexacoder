import base64

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def set_base64_encoded_name():
    '''
    This function extracts 'first_name' from the posted data,
    decodes