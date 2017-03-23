import psycopg2
import os
import json
import random
import re

dir = os.path.dirname(__file__)
configpath = os.path.join(dir, "../WebScraper/dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
cursor = connection.cursor()

# getbatches will only return articles with more than this many words
NUM_WORDS_THRESHOLD = 100

# getbatches will only return articles with less than this proportion of unknown words
PROPORTION_UNKOWN_THRESHOLD = 0.2

# This is just a magic number
SHUFFLE_DISTANCE = 10000


def getbatches(batchsize):
    """
    A generator which returns the normalized text and the appropriate labels in batches of
    size batchsize.
    :param batchsize: Number of articles to return at once
    :return: A generator for these batches. It yields lists of tuples, where each list is of length batchsize,
    and each tuple in the list is (content, valid), where content is the normalized text, and valid is a boolean
    representing whether this article is true.
    """
    cursor.execute("SELECT id FROM articles_normalized "
                   "WHERE unknown_words / num_words::FLOAT < %s AND num_words > %s ORDER BY num_words ASC",
                   (PROPORTION_UNKOWN_THRESHOLD, NUM_WORDS_THRESHOLD))
    results = [x for i, x in sorted(enumerate(cursor.fetchmany(200000)),
                                    key=lambda i: i[0] + (SHUFFLE_DISTANCE + 1) * random.random())]
    results = [id[0] for id in results]

    batch = []
    for id in results:
        batch.append(id)
        if len(batch) >= batchsize:
            cursor.execute("SELECT an.content, s.valid FROM articles_normalized an JOIN sources s ON an.source=s.url "
                           "WHERE id = ANY(%s)", (batch,))
            batch = []
            yield cursor.fetchmany(batchsize)

count = 0
for batch in getbatches(100):
    count += 1
    if count % 10 == 0:
        print(count)
