"""
This script is just to upload the list of fake news
to the database.
"""
import os
import psycopg2
import json
import re

dir = os.path.dirname(__file__)
configpath = os.path.join(dir, "../dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
connection.autocommit = True

listpath = os.path.join(dir, "FakeNewsList/fakenews")


def removewww(url):
    return re.sub(r'^((?:https?://)?)(ww(?:w|[0-9]+)\.)?(.*?)$', r'\1\3', url, flags=re.IGNORECASE)

cursor = connection.cursor()

with open(listpath) as listfile:
    for line in listfile:
        # line = listfile.readline()
        line = line[8:-1]
        line = removewww(line)
        try:
            cursor.execute("INSERT INTO sources (url, valid) VALUES (%s, %s);", (line, "false"))
        except psycopg2.IntegrityError:
            print("Url " + line + " already exists somehow.")
