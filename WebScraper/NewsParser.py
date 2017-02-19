
from newspaper import Article
from newspaper import fulltext
from newspaper import nlp
from newspaper.configuration import Configuration
from newspaper.extractors import ContentExtractor
import psycopg2
import requests
import json

with open("dbsettings.json") as configFile:
    config = json.load(configFile)
connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])

def OnArticleProcessError(url):
    print('Error occured when parsing url ' + url)

def StoreToDatabase(url, title, authors, text, keywords, summary):
    #Code to insert data to database
    print('Added :\n\t' + title + ' \n\t ' + str(authors) + ' \n\t ' + text[:100] + ' \n\t ' + str(keywords) + ' \n\t ' + summary)

def ProcessArticle(urlStr, htmlStr):
    config = Configuration()
    extractor = ContentExtractor(config)
    clean_doc = config.get_parser().fromstring(htmlStr)
    title = extractor.get_title(clean_doc)
    authors = extractor.get_authors(clean_doc)
    text = fulltext(htmlStr)

    text_keyws = list(nlp.keywords(text).keys())
    title_keyws = list(nlp.keywords(title).keys())

    keyws = list(set(title_keyws + text_keyws))
    summary_sents = nlp.summarize(title=title, text=text, max_sents=config.MAX_SUMMARY_SENT)
    summary = '\n'.join(summary_sents)

    if len(text) == 0 or len(authors) == 0:
        OnArticleProcessError(urlStr)
    else:
        StoreToDatabase(urlStr, title, authors, text, keyws, summary)

#test code for parsing html.
urlStr = 'https://www.nytimes.com/2017/02/18/us/politics/trump-candidates-top-posts.html'
testhtmlStr = requests.get(urlStr).text
ProcessArticle(urlStr, testhtmlStr)