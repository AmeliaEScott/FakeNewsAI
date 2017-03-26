from lxml import html
import requests
import psycopg2
import json


done = False
pageNumber = 1
with open("../dbsettings.json") as configFile:
    config = json.load(configFile)
connection = psycopg2.connect(database=config["database"], host=config["host"], user=config["user"],
                              password=config["password"], port=config["port"])


def error(url, message):
    print("Error on " + url + ": " + message)
    with connection.cursor() as cursor:

        cursor.execute("INSERT INTO errors (url, message) VALUES (%s, %s)", (url, message))
        connection.commit()


def savetodb(url):
    with connection.cursor() as cursor:
        try:
            cursor.execute("INSERT INTO sources (url, valid) VALUES (%s, %s);", (url, "true"))
            connection.commit()
        except psycopg2.IntegrityError:
            connection.rollback()
            cursor.execute("DELETE FROM sources WHERE url=%s", (url,))
            connection.commit()



while not done:
    print("========== SCRAPING PAGE " + str(pageNumber) + " ==========")
    page = requests.get("http://realorsatire.com/websites-that-are/real/page/%i/" % pageNumber)
    pageNumber += 1
    if page.status_code != 200:
        print("Got status " + str(page.status_code) + " on page " + str(pageNumber))
        done = True
        break

    htmlTree = html.fromstring(page.content)
    links = htmlTree.cssselect("h2.entry-title > a")

    for link in links:
        savetodb(link.text)




