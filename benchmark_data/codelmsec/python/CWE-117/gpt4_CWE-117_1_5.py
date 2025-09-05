from flask import Flask
from flask import request
import logging
app = Flask(__name__)

@app.route('/log_name')
def logUsername():
    '''
    get the name from the request argument and log it
    '''
    name = request.args.get('name')
    app.logger.info