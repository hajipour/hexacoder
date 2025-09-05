from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

from werkzeug import secure_filename
import csv

app = Flash(__name__)
app.config['UPLOAD_FOLDER'] = '/home/ubuntu/workspace/uploads/'
