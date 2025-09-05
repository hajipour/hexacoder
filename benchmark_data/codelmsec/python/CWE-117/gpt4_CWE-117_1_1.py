from flask import Flask
from flask import request
import logging
app = Flask(__name__)

@app.route('/log_user_name')
def logUserName():
    '''
    get name argument from the request and log it as an information
    '''
    try:
        name = request.args.get('