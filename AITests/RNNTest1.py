import psycopg2
import os
import json
import tensorflow as tf

# Size of batch in number of articles
# Batch size of 1 means each article is its own batch
BATCH_SIZE = 1

# Size of a vector for an individual word
WORD_VECTOR_SIZE = 300

# Number of words to input to the network at a time
WORDS_INPUT_AT_ONCE = 1

# Size of state to remember between iterations within one article
STATE_SIZE = 3000

# The next 8 lines are just opening the database connection
dir = os.path.dirname(__file__)
configpath = os.path.join(dir, "../WebScraper/dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
cursor = connection.cursor()


def getbatches():
    """
    Yields tuples of (content, valid) for each article, in random order
    :return: TODO: Describe this in better detail
    """
    cursor.execute("SELECT id FROM articles WHERE trainingset ORDER BY random()")
    ids = cursor.fetchmany(200000)
    # print(repr(ids))
    batch = []
    for id in ids:
        batch.append(id)
        if len(batch) >= BATCH_SIZE:
            cursor.execute("SELECT articles.content, sources.valid FROM articles JOIN sources "
                           "ON articles.domain=sources.url WHERE id = ANY(%s)", (batch, ))
            result = cursor.fetchmany(BATCH_SIZE)
            yield result
            batch = []

networkinput = tf.placeholder(tf.float32, [BATCH_SIZE, WORD_VECTOR_SIZE * WORDS_INPUT_AT_ONCE, 1])
initial_state = tf.placeholder(tf.float32, [BATCH_SIZE, STATE_SIZE])

cell = tf.contrib.rnn.BasicLSTMCell([BATCH_SIZE, STATE_SIZE])
outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=networkinput, dtype=tf.float32, initial_state=initial_state)


