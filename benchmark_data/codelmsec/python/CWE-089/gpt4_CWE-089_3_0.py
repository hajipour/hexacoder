from django.conf.urls import url
from django.db import connection

def display_user_info(username):
    '''
    Retrieves user information given their username.
    Extracts from the 'users' database table all data about a particular user.
    Sends SQL command "SELECT" via cursor execution. 
