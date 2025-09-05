from flask import Flask, request, make_response

from jinja import Environment, FileSystemloader
app = Flask(_name_)
loader = FileSystemloader()
env = Environment(autoescape=True, loader=loader)

def render(env