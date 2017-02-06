# -*- coding: utf-8 -*-
import scrapy
import re
from urllib.parse import urlparse

visited = set()


class NewsSpider(scrapy.Spider):
    name = "news"
    # allowed_domains = ["huffingtonpost.com"]
    start_urls = ['http://realorsatire.com/websites-that-are/real/',
                  'http://realorsatire.com/websites-that-are/satire/']

    # This function parses realorsatire.com to find the news sites to parse
    def parse(self, response):
        links = response.css("h2 > a::text").extract() # Find all links to outside news sites on this page
        for link in links:
            print("Starting to scrape " + link)
            # TODO: Determine if url is real or satire, and find a way to pass that info along to the callback
            yield scrapy.Request(url="http://" + link, callback=self.parsenewssite)

        match = re.match(r'^.*/page/([0-9]+)/$', response.url).group(1) # Find the page number in the URL
        if match is None: # No match is found on the first page, because the page number isn't in the URL
            pagenum = 1
        else:
            pagenum = int(match.group(1))
        pagenum += 1

        # This regex removes the page number from the url, if it's there, or just doesn't change anything otherwise
        nextpageurl = re.match(r'(^.*?)(/page/[0-9]+/)?$', response.url).group(1) + "/page/" + str(pagenum) + "/"
        print("Moving on to page " + nextpageurl)
        yield scrapy.Request(url=nextpageurl, callback=self.parse)

    # This function parses the pages of the actual news site
    def parsenewssite(self, response):
        uri = urlparse(response.url)
        domain = uri.netloc # The domain name of the current page

        # If this URL looks like an article...
        if re.match(r'.*?((?:[a-z0-9_]+)(?:-[a-z0-9_]+)+)/?$', uri.path, flags=re.IGNORECASE):
            # TODO: Find contents of article (inside <p> tags), determine if it is long enough to be considered an
            #  actual article, then yield it. (Along with other data, like the domain and whatnot)
            yield {
                'url': response.url
            }

        # Find all links on the page that go to the same URL that we're currently on
        # (We don't want to follow links to ads or other sites or whatever)
        links = response.css("a[href*='" + uri.netloc + "']::attr(href)").extract()
        for link in links:
            newuri = urlparse(link)
            # Specifically remove anything in the url that's a parameter or something like that, for reasons
            # (Many links have a bunch of parameters used by the site for tracking, so it makes it difficult
            # to keep track of which URLs have already been visited. So we remove all the parameters)
            url = "{uri.scheme}://{uri.netloc}{uri.path}".format(uri=uri)

            # TODO: Maybe store the visited URLs in the database instead of in memory?
            if url not in visited and domain == newuri.netloc:
                visited.add(url)
                yield scrapy.Request(url=url, callback=self.parsenewssite)
