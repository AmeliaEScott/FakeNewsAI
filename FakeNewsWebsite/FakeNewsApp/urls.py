from django.conf.urls import url
from . import views
from django.conf import settings
from django.conf.urls.static import static

"""
This fils is responsible for mapping URLs to Views.
A View is a python function that is executed when someone requests a specific URL, which returns
some sort of data. It could be HTML, or JSON, or XML, or whatever else.

For each url in the list below, the first argument is a regex for matching URLs. Django automatically cuts
off the domain of the url, so you're just left with what's after the first slash. For example,
if you visit http://website.com/something/url, Django gives you 'something/url'.

The second argument is the function to be executed when this URL is requested. For example, the following:
url(r'^', views.testview)
Maps the URL 'http://website.com/' to the function testview in the python file views.py.
"""

urlpatterns = [
    url(r'^$', views.index),
    url(r'^text/?$', views.testview),
    url(r'^judgearticle/?', views.judgearticleview),
    url(r'^judgetext/?', views.judgetextview),
]
