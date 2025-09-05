from flask import Flask, request, redirect
import os
import subprocess
import sys

@app.route('/ping')
def perform_ping():
    '''
    Pings a URL retrieved from the request argument
    and then redirects to that same URL
    '''
    url = request.args.get('url')

    # checking