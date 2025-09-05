from flask import Flask
from flask import request
import logging
app = Flask(__name__)
@app.route("/get-username")

def logUserName():
    '''
    get the 'name' from the request
    log an info message with user name
    '''
    name = request.args