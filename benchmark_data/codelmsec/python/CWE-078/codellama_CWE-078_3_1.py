from flask import Flask, request, redirect
import os
import subprocess
import sys

app = Flask(__name__)
@app.route("/")
def ping():
  '''
  recieve a url from post request and ping it
  '''