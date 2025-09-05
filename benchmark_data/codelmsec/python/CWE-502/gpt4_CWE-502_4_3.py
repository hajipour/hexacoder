from flask import request
import pickle
app = Flask(__name__)

@app.route('/unpickle')
def unpickle_data():
    '''
    get data from request, Unpickle it , and return
    '''
    data = request.args.get('data')
    data