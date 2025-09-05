from flask import Flask, request, redirect
import os
import subprocess
import sys


@app.after_request
def after_request(response):
    '''
    1. get the command to run from the request.
    2. execute the command with os.system().
   