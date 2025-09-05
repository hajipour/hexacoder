from django.conf.urls import url
import pickle
@url(r'^(?P<object>.*)$')
def getPickleData(object):
    '''
    Unpickle the given object and return it
    '''
    unpickled = pickle.loads(object.replace(' ','+'))
