import os
import psycopg2
import json
import re
from gensim.models import KeyedVectors

"""
This script is just for taking in data and normalizing it.
See the comments of normalize(text) for what this means.
"""


# The next 8 lines are just opening the database connection
dir = os.path.dirname(__file__)
configpath = os.path.join(dir, "../WebScraper/dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
connection.autocommit = True

cursor = connection.cursor()

dir = os.path.dirname(__file__)
modelpath = os.path.join(dir, "GoogleNews-vectors-negative300.bin")

# This step takes a VERY LONG TIME (1-2 minutes) and LOTS of memory (5 GB), so only do it once in any program!
model = KeyedVectors.load_word2vec_format(modelpath, binary=True)

# Number of articles to fetch at once. This number isn't very important
BATCH_SIZE = 100

# Number of total words an article must have to be considered an actual article
# (A value of 0 means to ignore this)
WORD_COUNT_THRESHOLD = 0

# The article must have less than this proportion of unknown words
# (A value of 1 basically means to ignore this)
WORD_PROPORTION_THRESHOLD = 1


def normalize(text):
    """
    Normalizes the text by doing the following:
    1. Remove all punctuation, except apostrophes
    2. Replace all whitespace with spaces
    3. Replace any multiple consecutive spaces with single spaces
    4. Convert all words to lowercase
    5. Replace all digits with #
    6. Replace any words not present in the language model with <UNK>

    :param text: Text to normalize
    :return: Three results: numwords, numunknown, text. Numwords is the number of words (including <UNK>),
             numunknown is the number of unknown words, and text is the normalized text
    """

    text = text.lower()

    # Replace all whitespace with single spaces
    text = re.sub("\s+", " ", text)

    # Replace all digits with number sign
    text = re.sub("\d", "#", text)

    # Remove anything that's not a letter, number sign, apostrophe, or space
    text = re.sub("[^a-z#' ]", "", text, flags=re.IGNORECASE)

    words = text.split(" ")
    words_out = []

    totalwords = len(words)
    unknownwords = 0

    for word in words:
        if word in model:
            words_out.append(word)
        else:
            unknownwords += 1
            words_out.append("<UNK>")
    return totalwords, unknownwords, " ".join(words_out)


cursor.execute("SELECT id, content FROM articles WHERE id NOT IN (SELECT id FROM articles_normalized) "
               "ORDER BY id LIMIT %s",
               (BATCH_SIZE, ))
results = cursor.fetchmany(BATCH_SIZE)
batchnumber = 1

while len(results) > 0:

    for result in results:
        numwords, numunknown, normaltext = normalize(result[1])
        # print("ID: %d, total words: %d, unknown words: %d" % (result[0], numwords, numunknown))
        if numwords > WORD_COUNT_THRESHOLD and numunknown / numwords < WORD_PROPORTION_THRESHOLD:
            # print(normaltext)
            cursor.execute("INSERT INTO articles_normalized (id, content, num_words, unknown_words, training_set) "
                           "VALUES (%s, %s, %s, %s, %s)", (result[0], normaltext, numwords, numunknown, True))
        # print("\n")
    print("Finished article number %d" % (batchnumber * BATCH_SIZE))

    cursor.execute("SELECT id, content FROM articles WHERE id NOT IN (SELECT id FROM articles_normalized) "
                   "ORDER BY id LIMIT %s",
                   (BATCH_SIZE, ))
    batchnumber += 1
    results = cursor.fetchmany(BATCH_SIZE)
