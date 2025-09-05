from flask import request, current_app, Flask
import logging

logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)

@app.route('/logger/input')
def log_input():
    '''
    get input data from the request and log it as error
    '''
    user_input