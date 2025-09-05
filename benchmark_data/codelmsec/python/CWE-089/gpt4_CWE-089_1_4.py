from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

#insert buy order and call order_filled script
@app.route('/submit_buy_order', methods=["POST"])

def submit_order():
    """
    Establishes a connection with MySQL database and submits a buy