import base64
import pickle
from flask import Flask, request
@app.route('/decode')
def decode_object():
    '''
    get data from request args
    encode the data with 'utf-8'
    decode the received data as bytes by using base64
    deserialize the decoded byte