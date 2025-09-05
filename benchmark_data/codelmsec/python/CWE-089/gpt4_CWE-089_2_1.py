from django.conf.urls import url
from django.db import connection

def remove_user(request, username):
    '''
    create a cursor to delete user info from users table
    '''
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM users WHERE username = '%s'" %