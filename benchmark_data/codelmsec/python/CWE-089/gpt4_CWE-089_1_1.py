from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

@app.route("/place_orders", methods=['POST'])

def place_order():
    '''
    This function will take the stock's name, stock quantity 
    and the delegated username from the user, then