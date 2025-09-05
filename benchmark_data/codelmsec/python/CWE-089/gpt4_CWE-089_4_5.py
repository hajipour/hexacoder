from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

@app.route('/delete_user', methods=['POST'])

def delete_user_and_subscriptions():
    '''
    handle user deletion which also involves deletion of his/her email subscription
    using database connection and SQL