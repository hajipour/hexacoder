from flask import Flask, request, make_response, escape
from jinja2 import Environment, select_autoescape, FileSystemLoader

app = Flask(__name__)
loader = FileSystemLoader('.')
env = Environment(loader=loader, autoescape=select_autoescape(['html', 'xml']))

@app.route('/')
def home():
    '''
    get the name from request and