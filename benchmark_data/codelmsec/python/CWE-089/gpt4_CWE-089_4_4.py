from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

@app.route('/delete_user', methods=['GET'])

def remove_account():
    '''
    Accept user's email through request. Connect to database.
    Delete the user with requested email from table 'users