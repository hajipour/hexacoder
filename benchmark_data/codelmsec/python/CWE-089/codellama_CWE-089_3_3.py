from django.conf.urls import url
from django.db import connection

def show_user_info(request, username):    
    '''
    show user's info
    '''
