from django.conf.urls import url
import pickle
def getPickleData(request, object):
    '''
    Retrieve object from URL parameter and load the pickled object
    '''
    try:
        pickled = request.GET.get(object)
        defoliated = pickle.loads(p