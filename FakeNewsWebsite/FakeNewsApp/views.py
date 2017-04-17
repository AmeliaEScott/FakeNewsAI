from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import random

DEBUG = False
if not DEBUG:
    from .articleutils import *

# Create your views here.

TRUE_THRESHOLD = 0.51

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
        return render(request, "FakeNewsApp/judgearticle.html", context=judgearticle(url=url))


def judgetextview(request):
    text = request.POST.get("text", None)
    if text is None:
        return HttpResponse("No article text provided.")
    else:
        return render(request, "FakeNewsApp/judgearticle.html", context=judgearticle(text=text))


def judgearticle(url=None, text=None):
    """
    This function does the heavy lifting of inputting the article to the neural network and
    getting a result.
    :param url: URL of the article to judge.
    :param text: If provided, then url will be ignored, and this will be used as the article text.
    :return: A python dict containing 'url', 'verdict' (True or False), 'score' (0.0 - 1.0), 'content' (article text),
             'word_scores' (List containing score for each word of the article),
             or None if there was an error
    """
    if DEBUG:
        return {
            'url': url,
            'verdict': False,
            'score': 0.12,
            'content': 'This is a test article. If you are reading this, then debug mode is on.',
            'word_scores': [random.random() for _ in range(0, 15)]
        }
    if text is None:
        text = getarticletext(url)
        if text is None:
            return None

    numwords, textvector = texttovector(text)
    scores, averagescore = scorearticle(textvector=textvector, numwords=numwords)

    return {
        'url': url,
        'verdict': averagescore > TRUE_THRESHOLD,
        'score': averagescore,
        'content': text,
        'word_scores': scores
    }
