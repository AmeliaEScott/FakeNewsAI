
from newspaper import fulltext
from newspaper import nlp
from newspaper.configuration import Configuration
from newspaper.extractors import ContentExtractor
import psycopg2
import json
import signal
import sys

batchsize = 50

with open("../dbsettings.json") as configFile:
    config = json.load(configFile)
connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
connection.autocommit = True

keepgoing = True


def OnArticleProcessError(url):
    print('Error occured when parsing url ' + url)


def StoreToDatabase(url, domain, title, authors, text, keywords, summary, cursor):
    # Code to insert data to database
    # print('Added :\n\t' + title + ' \n\t ' + str(authors) + ' \n\t ' + text[:100] + ' \n\t ' + str(keywords) + ' \n\t ' + summary)
    print("Added %s" % title)
    cursor.execute("INSERT INTO articles (batch, url, content, domain, title, authors, keywords, summary) "
                   "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                   ("Test1", url, text, domain, title, authors, keywords, summary))


def ProcessArticle(urlStr, domain, htmlStr, cursor):
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

    if len(text) == 0:
        OnArticleProcessError(urlStr)
    else:
        StoreToDatabase(urlStr, domain, title, authors, text, keyws, summary, cursor)


def controlc(*args, **kwargs):
    global keepgoing
    keepgoing = False
    print("Stopping as soon as this batch is done")


query = ("UPDATE articles_visited a "
         "SET    processed = TRUE "
         "FROM  ( "
         "    SELECT url "
         "    FROM   articles_visited "
         "    WHERE  processed = FALSE "
         "    LIMIT  %s "
         "    FOR UPDATE "
         ") sub "
         "WHERE  a.url = sub.url "
         "RETURNING a.url, a.domain, a.html;")

signal.signal(signal.SIGINT, controlc)

with connection.cursor() as cursor:
    cursor.execute(query, (batchsize, ))
    results = cursor.fetchmany(batchsize)
    while results is not None and keepgoing:
        for row in results:
            try:
                ProcessArticle(row[0], row[1], row[2], cursor)
            except AttributeError:
                print("What the fuck")
        if keepgoing:
            cursor.execute(query, (batchsize,))
            results = cursor.fetchmany(batchsize)
