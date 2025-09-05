from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(name)

#check to see if the user is in the database and insert the stock name and quantity into the buy_orders table
@app.route("/buy_stock")
