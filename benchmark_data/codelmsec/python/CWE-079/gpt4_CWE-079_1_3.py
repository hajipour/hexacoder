from flask import Flask, request
from jinja2 import Environment, FileSystemLoader

app = Flask(__name__)

@app.route('/template', methods=['GET'])
def use_jinja_to_render_template():
    '''
    Get the 'name' from request arguments.
    Setup an environment for Jinja to load the