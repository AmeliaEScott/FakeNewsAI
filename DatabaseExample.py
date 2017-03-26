import os
import json
import psycopg2

scriptdir = os.path.dirname(__file__)
configpath = os.path.join(scriptdir, "WebScraper/dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])

with connection.cursor() as cursor:
    cursor.execute("SELECT content, title, authors, keywords, summary FROM articles ORDER BY random() LIMIT 1;")
    result = cursor.fetchone()
    # print(str(result))
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    print("Title: " + result[1])
    print("authors: " + str(result[2]))
    print("Keywords: " + str(result[3]))
    print("Summary: " + result[4])
    print("Content: " + result[0])
