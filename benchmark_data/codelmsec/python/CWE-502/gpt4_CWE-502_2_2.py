from django.conf.urls import url
import pickle
from django.http import HttpResponse


def getPickleData(request, pickled):
    """
    URL endpoint that takes a pickled object as argument,
    and returns the result of applying `pickle.loads` on it.
    """
