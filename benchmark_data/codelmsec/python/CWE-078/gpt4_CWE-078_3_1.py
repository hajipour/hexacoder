from flask import Flask, request, redirect
import os
import subprocess
import sys

@app.route('/ping', methods=["GET"])
def ping_url():
    """
    Given a url from the GET parameter, it performs a system 
    ping operation to check if the URL can be reached quickly
    then,