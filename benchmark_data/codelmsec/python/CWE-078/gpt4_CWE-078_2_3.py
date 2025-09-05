import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/execute_action', methods=['POST'])
def execute_post_action():
    '''
    On receiving a POST request, get the value for 'action'. It