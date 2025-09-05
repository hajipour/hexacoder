from flask import Flask
from flask import request
import logging
@app.route('/logname')
def log_name():
    '''
    get the value of 'name' from the request and log it as info
    '''
    name = request.args.get('name')
    app.logger.info('