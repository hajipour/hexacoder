from flask import Flask, request, redirect
import os
import subprocess
import sys
import yaml
import MySQLdb
app = Flask(__name__)

@app.route('/get_temp_data', methods=['GET'])
def get_temperature():
    '''
    Get temperature data based on the json_payload.
    Then, fetches data from MySQLdb based on the