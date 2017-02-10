import os
import psycopg2
import json
import scrapy

dir = os.path.dirname(__file__)
configpath = os.path.join(dir, "../../dbsettings.json")
with open(configpath) as configFile:
    config = json.load(configFile)

connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])
connection.autocommit = True


class DBQueue:

    def __init__(self):
        # print("INITIALIZING SPECIAL QUEUE...")
        self.cursor = connection.cursor()
        self.size = 0

    def push(self, item):
        priority = item.priority
        # print("Inserting with priority %d" % priority)

        self.cursor.execute("SELECT count(1) FROM queue WHERE url=%s", (item.url, ))
        exists = self.cursor.fetchone()[0]

        if exists == '0':
            self.cursor.execute("INSERT INTO queue (priority, url, meta) VALUES (%s, %s, %s)",
                            (priority, item.url, json.dumps(item.meta)))
            self.size += 1

    def pop(self):
        self.cursor.execute("SELECT id, priority, url, meta FROM queue ORDER BY priority desc, id asc LIMIT 1")
        result = self.cursor.fetchone()
        # print(repr(result))
        print("Deleting id %d" % result[0])
        self.cursor.execute("DELETE FROM queue WHERE id=%s", (result[0],))
        self.size -= 1
        return scrapy.Request(url=result[2], priority=int(result[1]), meta=json.loads(result[3]))

    def close(self):
        self.cursor.close()

    def __len__(self):
        self.cursor.execute("SELECT count(*) FROM queue;")
        return int(self.cursor.fetchone()[0])
