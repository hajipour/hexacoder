from django.conf.urls import url
from django.db import connection

# Define a Django view that handles URL requests and fetches details of specified user

def fetch_user_details(username):
    '''
    Fetch details for the specified user from 'users' table in the database and return those as HTTP