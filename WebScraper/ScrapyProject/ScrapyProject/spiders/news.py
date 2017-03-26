# -*- coding: utf-8 -*-
import scrapy
import re
import os
from urllib.parse import urlparse, urljoin
import json
import psycopg2
import logging

targetarticlesperdomain = 1000  # Number of articles to try and get for each domain

scriptdir = os.path.dirname(__file__)
configpath = os.path.join(scriptdir, "../../../../dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
connection.autocommit = True

"""
To change this cralwer to another website, just change the following:
 - urlstoignore: List of regexes for URLs that should not be counted as articles
 - urlsnottofollow: URLs that don't lead to normal web pages (E.g., images, PDFs)
 - articleregex: Regex to match URLs of pages that are probably articles
 - starturl: URL at which to start crawling
"""

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
    re.compile(r'/blogs?/', flags=re.IGNORECASE),  # Blogs on news sites are apparently a thing
]

# URLs that should not be followed by the crawler
urlstonotfollow = [
    re.compile(r'(\.png|\.gif|\.jpe?g)$', flags=re.IGNORECASE),  # Images
    re.compile(r'\.pdf$', flags=re.IGNORECASE),  # PDFs
    # re.compile(r'/opinions?/', flags=re.IGNORECASE) # There's a LOT of opinion pages on CNN
    # re.compile(r'.*?bbc.com/[^n](?:[^e][^w][^s])?'),  # Ignore everything that's not in the news category
    # re.compile(r'.*?foxnews.com/opinion', flags=re.IGNORECASE)
]

# URLs that SHOULD be followed by the crawler
# The crawler only follows URLs that match at least one regex in this list, and
# match NONE of the regexes in urlstonotfollow
urlstofollow = [
    # re.compile(r'.*?foxnews\.com/(?:us|politics)', flags=re.IGNORECASE),
    re.compile(r'.*?usatoday.com.*?(?:washington|politics)', flags=re.IGNORECASE)
]

# Regex that a URL should match to be considered an article
# This one is for CNN. It matches articles from 2014-2017, in the categories 'us' or 'politics'
# articleregex = re.compile(r'.*?cnn.com/201[4-7]/[0-9]{2}/[0-9]{2}/(us|politics)/[a-z0-9_\-]*/index\.html',
#                           flags=re.IGNORECASE)
# articleregex = re.compile(r'.*?foxnews\.com/(?:us|politics)/201[4-7]/[0-9]{2}/[0-9]{2}/', flags=re.IGNORECASE)
articleregex = re.compile(r'.*?usatoday.com/story/news/politics/201[4-7]/[0-9]{2}/[0-9]{2}', flags=re.IGNORECASE)

starturl = 'http://www.usatoday.com'
startdomain = 'usatoday.com'


class NewsSpider(scrapy.Spider):

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
        return articleregex.match(url) is not None

    @staticmethod
    def shouldfollow(url):
        shouldfollow = False
        for regex in urlstofollow:
            if regex.search(url) is not None:
                shouldfollow = True
                break
        if not shouldfollow:
            # print("Not following " + url)
            return False
        for regex in urlstonotfollow:
            if regex.search(url) is not None:
                return False
        # print("Yes following " + url)
        return True

    def start_requests(self):
        yield scrapy.Request(url=starturl, callback=self.parse, meta={'domain': startdomain})
        # with connection.cursor() as cursor:
        #     cursor.execute("SELECT url, valid FROM sources WHERE valid = 'true'")
        #
        #     result = cursor.fetchmany(10000)
        #     for url in result:
        #         print(url[0])
        #     for url in result:
        #         logging.info("Yielding start url %s" % url[0])
        #         yield scrapy.Request(url="http://" + url[0], callback=self.parse,
        #                              meta={'domain': url[0], 'valid': url[1].lower() == 'true'},
        #                              priority=targetarticlesperdomain + 1)

    def parse(self, response):

        # print("Parsing " + response.url)

        currenturi = urlparse(response.url)
        currentdomain = currenturi.netloc  # The domain name of the current page

        domainwithoutwww = self.removewww(currentdomain)

        cleanedurl = "{uri.scheme}://{uri.netloc}{uri.path}".format(uri=currenturi)

        if response.meta.get("domain") != self.removewww(currentdomain):
            print("Domains changed. Expected: " + response.meta.get("domain") + ", got: " + response.url)
            return  # This happens when we were redirected (probably because of an ad)

        with connection.cursor() as cursor:

            # If this URL looks like an article...
            if self.isarticle(cleanedurl):
                # articles_visited is a child of visited, so everything inserted into articles_visited
                # is also in visited.
                try:
                    cursor.execute("INSERT INTO articles_visited (url, domain, html) VALUES (%s, %s, %s)",
                                   (cleanedurl, domainwithoutwww, response.text))
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
                    return

            cursor.execute("SELECT count(1) FROM visited WHERE domain=%s;", (domainwithoutwww,))
            count = cursor.fetchone()[0]
            priority = targetarticlesperdomain - int(count)

            # Find all links on the page that go to the same URL that we're currently on
            # (We don't want to follow links to ads or other sites or whatever)
            links = response.css("a::attr(href)").extract()
            for link in links:
                if currentdomain not in link:
                    # print("Adding domain to " + link)
                    link = response.urljoin(link)
                if not link.startswith("http"):
                    if link.startswith("://"):
                        link = "http" + link
                    else:
                        link = "http://" + link
                # print("About to follow link: " + link)
                newuri = urlparse(link)
                # Specifically remove anything in the url that's a parameter or something like that, for reasons
                # (Many links have a bunch of parameters used by the site for tracking, so it makes it difficult
                # to keep track of which URLs have already been visited. So we remove all the parameters)
                if newuri.netloc is None:
                    newuri.netloc = 'http'
                newurl = "{uri.scheme}://{uri.netloc}{uri.path}".format(uri=newuri)

                cursor.execute("select 1 from queue where url=%s union select 1 from visited where url=%s",
                               (newurl, newurl))
                result = cursor.fetchone()
                unvisited = result is None or len(result) == 0 or result[0] != 1
                if unvisited and currentdomain == newuri.netloc and self.shouldfollow(newurl):
                    yield scrapy.Request(url=newurl, callback=self.parse, meta=response.meta, priority=priority)
