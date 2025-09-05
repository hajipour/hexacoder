from flask import request
import pickle

app = Flask()

@app.get('/pickle')
def get_pickle_data():
    '''
        get pickled data and return it
    :return:
    '''