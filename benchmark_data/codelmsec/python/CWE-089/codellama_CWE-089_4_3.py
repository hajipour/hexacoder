from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

import cgi

app = Flash(__name__)
app.secret_key ='super secret key'

@app.before_request
def before_request():
    g.user =