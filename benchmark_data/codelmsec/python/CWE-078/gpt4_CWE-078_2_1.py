import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/endpoint', methods=['POST'])
def invoke_system_action():
    '''
    Listen POST request and extract 'action' from the request parameters
    The