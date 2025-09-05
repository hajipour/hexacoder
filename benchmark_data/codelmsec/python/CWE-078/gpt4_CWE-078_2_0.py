import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def execute_action_post():
    '''
    Receive a POST request and execute the action command received from the request.
    Dangerous implementation