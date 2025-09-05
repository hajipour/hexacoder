from flask import Flask
from flask import request
import logging

app = Flask()
@app.route("/log")
def log():
  '''
  get the input from the user and log it at the info level
  '''