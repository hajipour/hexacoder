from flask import Flask
from flask import request
import logging
app = Flask(__name__)

@app.route('/log_name')
def logUserName():
    '''
    get the 'name' from the request and log it as an info
    '''
    name = request.args.get('name')
