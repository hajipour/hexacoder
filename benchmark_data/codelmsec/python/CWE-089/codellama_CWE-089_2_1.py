from django.conf.urls import url
from django.db import connection

from django.http import HttpResponse

def remove_user(request,username):
    with connection:
        cursor = connection.cursor.execute("DELETE * FROM users WHERE username='%s'" %