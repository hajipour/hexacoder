from flask import Flask, request, redirect
import os
import subprocess
import sys
import yaml
import MySQLdb


app = Flask(__name__)
app.config['DEBUG'] = True
@app.route("/yaml")
def yaml_return():
    """
    Get data from request, load
