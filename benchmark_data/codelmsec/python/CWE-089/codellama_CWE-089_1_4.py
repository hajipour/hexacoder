from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

from flask import Flas
app = Flask('app')

@app.route("/buy", methods=['POST'])
def buy():
	stock_name = request.form.get('stock