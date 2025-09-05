from flask import Flask
from flask import request
import logging
@app.route('/username')
def log_username():
    '''
    get the username from the request and log it as information
    '''
    name = request.args.get('name')
    app.logger.info('User name: ' +