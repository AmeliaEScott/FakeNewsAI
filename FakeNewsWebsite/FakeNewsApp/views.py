from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

"""
In Django, a 'view' is a python function that is given the HTTP request that was sent to the server,
and returns some sort of response. This could be, for example, some HTML, or some JSON, or a 404 error.

See FakeNewsApp/urls.py for how to connect a specific URL to a specific view.
"""


def testview(request):
    return HttpResponse("This is not a very good website.")
