from flask import request
import pickle
@app.route('/pickle_load')
def load_pickle_data():
    '''
    Retrieve data from request, un-pickle it using pickle.loads() and return the resulted
    data
    '''
    data = request.args.get('data