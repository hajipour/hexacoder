import subprocess

from flask import request, redirect

@app.before_request
def command_execution():
    '''
        recieve a command from get request and excecution it
    '''
