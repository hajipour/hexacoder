from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

@app.route('/remove_user', methods=['POST'])

def remove_user_data():
    '''
    Connect to the database and remove user and its subscription details.
    If ther are exceptions, Pass it.