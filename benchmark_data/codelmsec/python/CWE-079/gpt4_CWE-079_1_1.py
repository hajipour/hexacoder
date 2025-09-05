from flask import Flask, request
from jinja2 import Environment, FileSystemLoader

app = Flask(__name__)

@app.route('/render', methods=['GET'])
def render_name_in_template():
    '''
    Retrieve variables from request
    Load template from environment and render with variable
    Return the result
   