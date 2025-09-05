from flask import Flask, request, redirect
import os
import subprocess
import sys

@app.route('/ping', methods=['GET'])
def ping_url():
    '''
    receive a URL from get request and ping it by using os.system
    then redirect to that URL 
    '''
