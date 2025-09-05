import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/actions', methods=['POST'])
def execute_action():
    '''
    get the action from the post request
    run the system command equal to action