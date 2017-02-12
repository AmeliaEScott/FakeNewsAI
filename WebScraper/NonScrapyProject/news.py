# -*- coding: utf-8 -*-
import re
import os
from urllib.parse import urlparse
import json
import psycopg2
import logging
from lxml import html

targetarticlesperdomain = 1000  # Number of articles to try and get for each domain

scriptdir = os.path.dirname(__file__)
configpath = os.path.join(scriptdir, "../dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
connection.autocommit = True

# URLs that are probably not articles
urlstoignore = [
    re.compile(r'/tags?/', flags=re.IGNORECASE),  # All URLs with "/tag/" or "/tags/" in them
    re.compile(r'/categor(y|ies)/', flags=re.IGNORECASE),  # All URLs with /category/ or /categories/
    re.compile(r'/topics?/', flags=re.IGNORECASE),
    re.compile(r'/comments?-page-[0-9]+/?$', flags=re.IGNORECASE),  # URLs that have "comment-page-<number>"
    re.compile(r'/page/[0-9]+/?$', flags=re.IGNORECASE),  # URLs that are just pages containing many articles
    re.compile(r'(\.png|\.gif|\.jpe?g)$', flags=re.IGNORECASE),  # Images (.png, .gif, .jpeg, .jpg)
    re.compile(r'/author/|/people/', flags=re.IGNORECASE),  # Pages that are just bios of authors
    re.compile(r'/watch/', flags=re.IGNORECASE),  # Pages that are just videos
]

# URLs that should not be followed by the crawler
urlstonotfollow = [
    re.compile(r'(\.png|\.gif|\.jpe?g)$', flags=re.IGNORECASE),  # Images
    re.compile(r'\.pdf$', flags=re.IGNORECASE),  # PDFs
]


class NewsSpider:

    name = "news"

    @staticmethod
    def removewww(url):
        return re.sub(r'^((?:https?://)?)(ww(?:w|[0-9]+)\.)?(.*?)$', r'\1\3', url, flags=re.IGNORECASE)

    """
    Returns True if this URL is (probably) an article
    Returns False if this URL is on the list of URLs to ignore
    (Like /tag/, or images)
    """
    @staticmethod
    def isarticle(url):
        for regex in urlstoignore:
            if regex.search(url) is not None:
                return False
        return re.match(r'.*?/((?:[a-z0-9_]+)(?:-[a-z0-9_]+)*)/?', url, flags=re.IGNORECASE) is not None

    @staticmethod
    def shouldfollow(url):
        for regex in urlstonotfollow:
            if regex.search(url) is not None:
                return False
        return True

    def start_requests(self):
        with connection.cursor() as cursor:
            cursor.execute("SELECT url, valid FROM sources WHERE valid = 'false' "
                           "AND url NOT IN (SELECT url FROM visited);")

            for url in cursor.fetchmany(10000):
                yield {
                    'url': url,
                    'priority': targetarticlesperdomain
                }

    def parse(self, response):

        currenturi = urlparse(response.url)
        currentdomain = currenturi.netloc  # The domain name of the current page

        domainwithoutwww = self.removewww(currentdomain)

        cleanedurl = "{uri.scheme}://{uri.netloc}{uri.path}".format(uri=currenturi)

        with connection.cursor() as cursor:

            # If this URL looks like an article...
            if self.isarticle(cleanedurl):
                # articles_visited is a child of visited, so everything inserted into articles_visited
                # is also in visited.
                try:
                    cursor.execute("INSERT INTO articles_visited (url, domain, html) VALUES (%s, %s, %s)",
                                   (cleanedurl, domainwithoutwww, response.content))
                except psycopg2.IntegrityError:
                    logging.warning("Error inserting \"%s\" into articles_visited: Already there.", cleanedurl)
                    return
                # TODO: Find contents of article (inside <p> tags), determine if it is long enough to be considered
                # an actual article, then yield it. (Along with other data, like the domain and whatnot)

                """
                This code is REALLY slow and a bit buggy, so it might be better to just store the HTML,
                and parse it later.
                """
                # article = Article(cleanedurl)          # Initializes the Article, but doesn't do anything
                # article.download(html=response.text)   # Doesn't actually redownload the page if you give it html
                # article.parse()                        # Must call parse() before extracting text
                #
                # if article.authors is None or len(article.authors) < 1:
                #     author = 0
                # else:
                #     author = article.authors[0]
                #
                # cursor.execute("INSERT INTO articles (batch, url, content, domain, title, author, html) "
                #                "VALUES (%s, %s, %s, %s, %s, %s, %s)", ("ScrapverV2", cleanedurl, article.text,
                #                                                        domainWithoutWWW, article.title, author,
                #                                                        response.text))

            else:
                try:
                    cursor.execute("INSERT INTO visited (url, domain) VALUES (%s, %s)",
                                   (cleanedurl, self.removewww(currentdomain)))
                except psycopg2.IntegrityError:
                    logging.warning("Error inserting \"%s\" into visited: Already there.", cleanedurl)

            cursor.execute("SELECT count(1) FROM visited WHERE domain=%s;", (domainwithoutwww,))
            count = cursor.fetchone()[0]
            priority = targetarticlesperdomain - int(count)

            # Find all links on the page that go to the same URL that we're currently on
            # (We don't want to follow links to ads or other sites or whatever)
            tree = html.fromstring(response.content)
            links = tree.cssselect("a")
            for link in links:
                # print("About to follow link: " + link)
                newuri = urlparse(link.attrib['href'])
                # Specifically remove anything in the url that's a parameter or something like that, for reasons
                # (Many links have a bunch of parameters used by the site for tracking, so it makes it difficult
                # to keep track of which URLs have already been visited. So we remove all the parameters)
                newurl = "{uri.scheme}://{uri.netloc}{uri.path}".format(uri=newuri)

                cursor.execute("select 1 from queue where url=%s union select 1 from visited where url=%s",
                               (newurl, newurl))
                result = cursor.fetchone()
                unvisited = result is None or len(result) == 0 or result[0] != 1
                if unvisited and currentdomain == newuri.netloc and self.shouldfollow(newurl):
                    yield {
                        'url': newurl,
                        'priority': priority
                    }
