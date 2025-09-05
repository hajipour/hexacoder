from django.http import HttpResponseRedirect
from django.shortcuts import redirect
from django.views import View
  
class RedirectTargetView(View):
    """
    This View gets the target url from the request and redirects to it
    """
    def get(self, request, *