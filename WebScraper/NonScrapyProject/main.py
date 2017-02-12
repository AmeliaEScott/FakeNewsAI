import requests
from lxml import html
from .news import NewsSpider
from .dbqueue import DBQueue
import signal

spider = NewsSpider()
queue = DBQueue()
keepGoing = True


def siginthandler(var1, var2):
    print("Stopping gracefully...")
    global keepGoing
    keepGoing = False

signal.signal(signal.SIGINT, siginthandler)

while keepGoing:
    data = queue.pop()
    if data is not None:
        print("Crawling " + repr(data))
        response = requests.get(data['url'])
        if response is not None and response.status_code == 200:
            # tree = html.fromstring(response.content)
            generator = spider.parse(response)
            for link in generator:
                queue.push(link)
