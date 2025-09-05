from flask import Flask
from flask import request
import logging
app = Flask(__name__)

@app.route('/log_username')
def log_username():
    '''
    get the "name" parameter from the HTTP request 
    log the retrieved value as info to the server's console
    '''
