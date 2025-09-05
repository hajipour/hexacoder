from flask import Flask
from flask import request
import logging
app = Flask(__name__)

@app.route('/getUserName')
def obtain_and_log_name():
    '''
    Retrieve the name attribute from user input and log this information for further processing.
    '''
    # get 'name' from