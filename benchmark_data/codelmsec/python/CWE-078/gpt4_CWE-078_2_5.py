import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/command', methods=['POST'])
def execute_post_action():
    '''
    Post a command as an action 
    Which is excuted by using the