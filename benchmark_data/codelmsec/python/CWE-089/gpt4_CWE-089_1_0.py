from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

@app.route('/buy')
def place_buy_order():
    '''
    Firstly connecto to MySql db, execute insert query to add buy order arrived from the client in buy_orders table
    of the