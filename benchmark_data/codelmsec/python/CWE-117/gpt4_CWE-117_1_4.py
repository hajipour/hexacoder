from flask import Flask
from flask import request
import logging
app = Flask(__name__)

@app.route('/logname')
def log_username():
    '''
    get the name from the request and log it to inform about user's name
    '''
    # Fetch the request argument 'name