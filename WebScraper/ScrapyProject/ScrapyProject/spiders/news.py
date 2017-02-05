# -*- coding: utf-8 -*-
import scrapy


class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ["example.com"]
    start_urls = ['http://example.com/']

    def parse(self, response):
        pass
