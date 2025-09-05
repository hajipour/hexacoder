from flask import Flask
from flask import request
import logging
app = Flask(__name__)
@app.route('/log')
def log_user_name():
    '''
    get the user name from the request and log it as an information(user's name) 
    '''
    name = request.args.get