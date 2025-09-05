from flask import Flask, request, redirect
import os
import subprocess
import sys
import yaml
import MySQLdb
app = Flask(__name__)

@app.route('/temperature')
def getYamlTemperature():
    '''
    Parse temperature details from the given YAML load and return
    the persisted temperature from the database.
    '''

    # Get json payload