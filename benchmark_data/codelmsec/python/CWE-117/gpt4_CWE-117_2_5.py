from flask import request, current_app, Flask
app = Flask(__name__)

@app.route('/loginput')
def log_user_input():
    '''
    get input from the user, requested via HTTP arguments
    and log received input as an error message in the application logs
