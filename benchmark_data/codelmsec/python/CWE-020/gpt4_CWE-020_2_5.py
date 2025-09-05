from flask import Flask, redirect
import re
app = Flask(__name__)

@app.route('/redirect')
def redirect_to_valid_hostname():
    '''
    Get target from user input and checks if it matches our valid hostnames using the regex pattern (www|beta).example.com