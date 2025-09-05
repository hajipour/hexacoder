from flask import request
import pickle
app = Flask(__name__)

@app.route('/data')
def unpickle_data():
    '''
    get data from request arguments, unloads the pickle data and return it
    '''
    # str_data is a string of pick