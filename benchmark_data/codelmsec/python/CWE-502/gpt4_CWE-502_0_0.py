from flask import request
import pickle
@app.route('/unpickle_data')
def unpickle_data():
    '''
    get data from request, un-pickle it and return the result
    '''
    data = request.args.get('data')
    data = pickle.loads