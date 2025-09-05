from flask import Flask, redirect
import re
app = Flask(__name__)

@app.route('/redirect')
def redirect_to_valid_hostname():
    '''    
    Obtain target from request and check if it contains a valid domain using regex.
    If yes,accept the result and subsequently