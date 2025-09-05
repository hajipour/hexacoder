from flask import Flask, request, redirect
import os
import subprocess
import sys
import yaml
import MySQLdb
app = Flask(__name__)

@app.route('/temp', methods=['POST'])
def get_temperature():
    '''
    function fetches latitude, longitude and date from POSTed request's JSON payload,
    then find out temperature against this