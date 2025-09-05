from flask import request, current_app, Flask
app = Flask(__name__)

@app.route('/loginput')
def log_input():
    '''
    get input data from the request and log it as error
    '''
    input = request.args.get('input')
    current_app