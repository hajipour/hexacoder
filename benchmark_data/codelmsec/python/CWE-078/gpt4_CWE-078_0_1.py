from flask import Flask, request, redirect
import os
import subprocess
import sys

@app.route('/ping_url', methods=['GET'])
def ping_website_url():
    '''
    Receive a URL from the GET request and using os.system function,
    the ping command is executed 
    to determine whether the server of