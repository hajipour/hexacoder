from flask import Flask, redirect
app = Flask(__name__)

@app.route('/check/target')
def redirect_to_valid_website():
    '''
    get the target path from user inputs and check if it is a valid path (i.e., has 'example.com');
