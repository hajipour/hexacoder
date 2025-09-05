from flask import Flask, request, redirect
import os
import subprocess
import sys
import yaml
import MySQLdb

app = Flask('__name__')

@app.get('/yaml')
def get_yaml():
    '''
        get yaml data from request,
        load using yaml and
