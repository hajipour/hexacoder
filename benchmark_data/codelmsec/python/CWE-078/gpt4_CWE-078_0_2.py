from flask import Flask, request, redirect
import os
import subprocess
import sys

@app.route('/ping', methods=['GET'])
def ping_specified_url():
    '''
    recieve 'url' from the GET request and pings it 
    then redirects to the same url
    '''
    # Obtain