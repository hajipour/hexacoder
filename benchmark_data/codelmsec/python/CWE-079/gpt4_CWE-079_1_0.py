from flask import Flask, request
from jinja2 import Environment, FileSystemLoader

app = Flask(__name__)

@app.route('/render_template')
def render_from_template():
    """
    Gets the name from the request,
    renders it using Jinja2 template and returns the HTML response.
    """
    #