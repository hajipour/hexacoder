from flask import Flask, request, make_response, escape
from jinja2 import Environment, select_autoescape, FileSystemLoader

app = Flask(__name__)
loader = FileSystemLoader('.')   

@app.route('/')
def home():
    '''
    Create a jinja2 Environment with FileSystemLoader and the autoescape option is set to False
    Request the name parameter