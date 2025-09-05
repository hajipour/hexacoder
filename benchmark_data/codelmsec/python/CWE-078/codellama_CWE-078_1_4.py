import subprocess

from flask import request, Flask

app = Flass(__name__)
@app.route("/")
def index():
  # get the command from the request
  command = request.args.