from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

@app.route('/delete_user', methods=['GET'])

def delete_user():
    '''
    This function runs when the context root /delete_user is hit. 
    An email address is passed as a GET request param where a connection