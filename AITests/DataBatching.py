import psycopg2
import os
import sys
import json
import random

"""
This file handles separating data into batches in a somewhat random order.
"""

dir = os.path.dirname(__file__)
configpath = os.path.join(dir, "../dbsettings.json")
try:
    with open(configpath) as configFile:
        config = json.load(configFile)
except FileNotFoundError:
    print("Didn't find dbsettings.json in %s" % configpath)
    sys.exit()

try:
    connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                                  password=config["password"], port=config["port"])
    cursor = connection.cursor()
except psycopg2.OperationalError:
    print("Error connecting to database. Check that all of the information in your dbsettings.json is correct.")
    sys.exit()

# getbatches will only return articles with more than this many words
NUM_WORDS_THRESHOLD = 100

# getbatches will only return articles with less than this proportion of unknown words
PROPORTION_UNKOWN_THRESHOLD = 0.2

# The larger this number is, the more shuffled the data will be, but that means
# that each batch will have more varied article length, which means more padding for the shorter articles
SHUFFLE_DISTANCE = 1000


def getbatches(batchsize):
    """
    A generator which returns the normalized text and the appropriate labels in batches of
    size batchsize.
    :param batchsize: Number of articles to return at once
    :return: A generator for these batches. It yields lists of tuples, where each list is of length batchsize,
    and each tuple in the list is (content, valid), where content is the normalized text, and valid is a boolean
    representing whether this article is true.
    """
    cursor.execute("SELECT id FROM articles_normalized WHERE training_set ORDER BY num_words ASC")

    # This line shuffles the results such that each result doesn't end up too far away from where it started.
    # Because the results are initially in order of number of words, this means that every article will
    # be surrounded by other articles with similar length, so necessary padding is minimized.
    # To make it more mixed up, increase the constant SHUFFLE_DISTANCE.
    # This code is inspired by (read: directly copied from) this StackOverflow answer:
    # http://stackoverflow.com/a/30784808/2364686
    results = [x for i, x in sorted(enumerate(cursor.fetchmany(200000)),
                                    key=lambda i: i[0] + (SHUFFLE_DISTANCE + 1) * random.random())]

    # For reasons, "results" currently looks like this: [(12,), (31974,), ... ]
    # This line just gets rid of the nested tuple crap and expands it into a flat list
    results = [id[0] for id in results]

    # At this point, results is just a normal list of IDs, like this: [1, 2, 3, 4, ...]
    # (They are sort of in order of number of words, so that batches have similar number of words per article)
    # Now, we split it up into batches:
    results = [results[i: i + batchsize] for i in range(0, len(results), batchsize)]
    # Now, results look like this: [ [1, 2, 3], [4, 5, 6] ... ]

    # Results can now be randomly shuffled without mixing articles of different lengths together
    random.shuffle(results)

    for ids in results:
        # This is the part where it actually retrieves the relevant content from the database
        cursor.execute("SELECT an.content, s.valid FROM articles_normalized an JOIN sources s ON an.source=s.url "
                       "WHERE id = ANY(%s)", (ids,))
        # "yield" is what makes this function a generator, so you can iterate over it using "for ... in"
        batch = cursor.fetchmany(batchsize)
        if len(batch) == batchsize:
            yield batch


if __name__ == "__main__":
    print("Running this code the way you're doing it now isn't very useful. This is just demo code.")

    # Here's just some sample code for how to use this function.
    # Currently, I just have it set up to show me how fast it goes through the dataset.
    # (Spoiler alert: Not very fast)
    # TODO: Look into downloading all of the data as, like, a CSV or something, to limit number of database calls
    count = 0
    for batch in getbatches(10):
        for article in batch:
            print("%d %s" % (len(article[0].split(" ")), article[1]))
        print("\n")
