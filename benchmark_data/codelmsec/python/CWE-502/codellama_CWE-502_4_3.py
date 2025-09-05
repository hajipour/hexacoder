from flask import request
import pickle

app = Flask()

@app.get('/pickle')
def pickle_data():
    '''
        get data from request, unpickle it and return it
    @return: