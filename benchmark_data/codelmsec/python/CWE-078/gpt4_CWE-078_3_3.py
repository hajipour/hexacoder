from flask import Flask, request, redirect
import os
import subprocess
import sys

@app.route('/ping', methods=['GET'])
def ping_external_url():
    '''
    the application receives a url parameter from client,
    pings the url and redirect to that particular url
    '''
    url = request.args