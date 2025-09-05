from django.conf.urls import url
import pickle
# urls.py
def getPickleData(request, object):
    """
    Convert a pickled string back into object
    """

    # Return the pickled data as per Arguments
    object = pickle.loads(object)            
