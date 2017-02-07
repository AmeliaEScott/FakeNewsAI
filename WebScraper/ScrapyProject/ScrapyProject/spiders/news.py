# -*- coding: utf-8 -*-
import scrapy
import re
import os
from urllib.parse import urlparse
import json
import psycopg2
from psycopg2 import pool


dir = os.path.dirname(__file__)
configpath = os.path.join(dir, "../../../dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
connection.autocommit = True

class NewsSpider(scrapy.Spider):

    name = "news"

    def removewww(self, url):
        return re.sub(r'^((?:https?://)?)(ww(?:w|[0-9]+)\.)?(.*?)$', r'\1\3', url, flags=re.IGNORECASE)

    def start_requests(self):
        with connection.cursor() as cursor:
            cursor = connection.cursor()
            cursor.execute("SELECT url, valid FROM sources;")

            for url in cursor.fetchmany(10000):
                yield scrapy.Request(url="http://" + url[0], callback=self.parse,
                                     meta={'domain': url[0], 'valid': url[1].lower() == 'true'})

    def parse(self, response):

        currenturi = urlparse(response.url)
        currentdomain = currenturi.netloc  # The domain name of the current page

        cleanedurl = "{uri.scheme}://{uri.netloc}{uri.path}".format(uri=currenturi)

        if response.meta.get("domain") != self.removewww(currentdomain):
            print("Domains changed. Expected: " + response.meta.get("domain") + ", got: " + response.url)
            return  # This happens when we were redirected (probably because of an ad)

        with connection.cursor() as cursor:

            cursor.execute("INSERT INTO visited (url, domain) VALUES (%s, %s)",
                           (cleanedurl, self.removewww(currentdomain)))

            # If this URL looks like an article...
            if re.match(r'.*?((?:[a-z0-9_]+)(?:-[a-z0-9_]+)*)/?$', currenturi.path, flags=re.IGNORECASE):
                # TODO: Find contents of article (inside <p> tags), determine if it is long enough to be considered
                # an actual article, then yield it. (Along with other data, like the domain and whatnot)
                yield {
                    'url': response.url,
                    'valid': response.meta.get("valid")
                }

            # connection.commit()

            # Find all links on the page that go to the same URL that we're currently on
            # (We don't want to follow links to ads or other sites or whatever)
            links = response.css("a::attr(href)").extract()
            for link in links:
                # print("About to follow link: " + link)
                newuri = urlparse(link)
                # Specifically remove anything in the url that's a parameter or something like that, for reasons
                # (Many links have a bunch of parameters used by the site for tracking, so it makes it difficult
                # to keep track of which URLs have already been visited. So we remove all the parameters)
                url = "{uri.scheme}://{uri.netloc}{uri.path}".format(uri=newuri)

                # TODO: Maybe store the visited URLs in the database instead of in memory?
                cursor.execute("SELECT count(1) FROM visited WHERE url=%s", (self.removewww(newuri.netloc), ))
                alreadyvisited = cursor.fetchone()[0] == '1'
                if not alreadyvisited and currentdomain == newuri.netloc:
                    yield scrapy.Request(url=url, callback=self.parse, meta=response.meta)
