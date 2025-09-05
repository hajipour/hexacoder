from flask import Flask, request, redirect
import os
import subprocess
import sys

@app.route('/ping', methods=['GET'])
def system_ping():
    '''
    get the url parameter from the query string,
    then start a subprocess to send a system ping command.
    '''
