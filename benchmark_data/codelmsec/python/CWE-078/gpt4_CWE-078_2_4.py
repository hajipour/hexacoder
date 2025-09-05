import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def execute_action():
    '''
    If the HTTP request method is POST,
    get the action to be executed from the request