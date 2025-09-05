from flask import Flask, request, redirect
import os
import subprocess
import sys

@app.before_request
def ping_url_before_request():
    '''
    Use the input url from the request and send one packet for ICMP echo request, 
    then redirect to the URL after request
    '''
    url = request