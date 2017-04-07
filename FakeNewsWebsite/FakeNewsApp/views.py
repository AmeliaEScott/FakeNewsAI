from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import random

# Create your views here.

"""
In Django, a 'view' is a python function that is given the HTTP request that was sent to the server,
and returns some sort of response. This could be, for example, some HTML, or some JSON, or a 404 error.

See FakeNewsApp/urls.py for how to connect a specific URL to a specific view.
"""


def testview(request):
    # The context fills in the template with the appropriate data.
    # Because we've provided a variabel 'num', every time {{ num }} appears in the template 'test.html',
    # it will be replaced with the number that we provided.
    return render(request, "FakeNewsApp/test.html", context={
        'num': random.randint(0, 100)
    })


def index(request):
    return render(request, "FakeNewsApp/index.html")


def judgearticleview(request):
    url = request.GET.get('url', None)
    if url is None:
        return HttpResponse("No url provided.")
    else:
        return render(request, "FakeNewsApp/judgearticle.html", context=judgearticle(url))


def judgearticle(url):
    """
    This function does the heavy lifting of inputting the article to the neural network and
    getting a result.
    :param url: URL of the article to judge
    :return: A python dict containing 'url', 'verdict' (True or False), 'score' (0.0 - 1.0), 'content' (article text),
             'word_scores' (List containing score for each word of the article)
    """

    return {
        'url': url,
        'verdict': True,
        'score': 0.3,
        'content': 'This is some long string that contains the text of the article.',
        'word_scores': [0.1, 0.9, 0.886, 0.1023, 0.5]
    }
