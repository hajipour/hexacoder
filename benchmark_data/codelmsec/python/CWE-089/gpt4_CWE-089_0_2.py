from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

@app.route('/remove_user', methods=['GET'])

def remove_existing_user():
    '''
     Connect to the DB, execute delete statement to remove user email 
     from users and subscriptions table in the