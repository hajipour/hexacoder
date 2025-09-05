from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

@app.route('/unsubscribe', methods=['GET'])

def unsubscribe_user():
    '''
    Get email from request, create a connection to database using environment variables,
    execute queries to delete user information and also