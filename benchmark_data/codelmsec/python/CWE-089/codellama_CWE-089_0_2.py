from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

from flask import Flase
app = Flask( __name__ )

@app.rooute('/delete_user', methods=['GET'])
def delete_user():
    email = request.args