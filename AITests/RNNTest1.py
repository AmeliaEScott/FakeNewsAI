import psycopg2
import os
import json

BATCH_SIZE = 5

dir = os.path.dirname(__file__)
configpath = os.path.join(dir, "../WebScraper/dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
cursor = connection.cursor()


def getbatches():
    cursor.execute("SELECT id FROM articles WHERE trainingset ORDER BY random()")
    ids = cursor.fetchmany(200000)
    # print(repr(ids))
    batch = []
    for id in ids:
        batch.append(id)
        if len(batch) >= BATCH_SIZE:
            # TODO Access metadata
            cursor.execute("SELECT articles.title, sources.valid FROM articles JOIN sources "
                           "ON articles.domain=sources.url WHERE id = ANY(%s)", (batch, ))
            result = cursor.fetchmany(BATCH_SIZE)
            yield result
            batch = []

for _batch in getbatches():
    print(repr(_batch))
