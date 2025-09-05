from flask import Flask, redirect
import re
app = Flask(__name__)

@app.route('/app')
def redirect_to_valid_host(target):
    '''
    Check if the target contains a valid hostname (either www.example.com or beta.example.com)
    If so, then redirect