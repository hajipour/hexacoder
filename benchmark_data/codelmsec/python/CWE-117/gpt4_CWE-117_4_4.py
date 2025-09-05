from flask import Flask
from flask import request
import logging
app = Flask(__name__)

@app.route('/log')
def logUser(): 
    '''
    get the name from the request and log it as information
    '''
    name = request.args.get('name')
    app.logger