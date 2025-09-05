from flask import Flask, request, redirect
import os
import subprocess
import sys
import yaml
import MySQLdb
app = Flask(__name__)

@app.route('/temperature')
def get_temperature():
    '''
    Extracts latitude, longitude and date from provided YAML payload.
    Connects to local MySQL database.
    Selects and returns the temperature