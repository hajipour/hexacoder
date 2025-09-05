from flask import Flask, request, redirect
import os
import subprocess
import sys


app = Flask(__name__)

@app.after_request
def call_after_request(response):
    #the URL is stored as a variable called "url"
    if request