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


def savetodb(url, quote, validity):
    data = {
        'url': url,
        'quote': quote,
        'valid': validity
    }
    # print(repr(data))
    with connection.cursor() as cursor:
        cursor.execute("INSERT INTO articles (batch, url, content, valid) VALUES (%s, %s, %s, %s);", ("snopes 1", url, quote, validity))
        connection.commit()


def scrapepage(url):
    print("Scraping URL " + url)
    page = requests.get(url)
    if page.status_code != 200:
        error(url, "Unexpected error code: " + str(page.status_code))
        return
    htmlTree = html.fromstring(page.content)
    quotes = htmlTree.cssselect("div[itemprop=reviewBody] > blockquote")
    if len(quotes) <= 0:
        error(url, "No blockquote found")
        return

    quote = quotes[0]
    quotetext = ""
    for quoteParagraph in quote.cssselect("*"):
        if quoteParagraph.text is not None:
            quotetext += quoteParagraph.text + "\n"
    # print(quoteText + "\n\n\n")

    validitytext = htmlTree.cssselect("span[itemprop=reviewRating] > span[itemprop=alternateName]")[0].text
    validityimage = htmlTree.cssselect("div.claim-old > img")[0]

    if validitytext is None and validityimage is None:
        error(url, "Could not find verdict image or text")
        return

    if (validitytext is not None and validitytext.lower() == 'false') or (validityimage is not None and "red" in validityimage.attrib['src'].lower()):
        validity = "false"
    elif (validitytext is not None and validitytext.lower() == 'true') or (validityimage is not None and "green" in validityimage.attrib['src'].lower()):
        validity = "true"
    elif (validitytext is not None and validitytext.lower() == 'mixture') or (validityimage is not None and "mix" in validityimage.attrib['src'].lower()):
        validity = "mix"
    elif (validitytext is not None and validitytext.lower() == 'mostly false') or (validityimage is not None and "mostlyfalse" in validityimage.attrib['src'].lower()):
        validity = "mostly false"
    elif (validitytext is not None and validitytext.lower() == 'mostly true') or (validityimage is not None and "mostlytrue" in validityimage.attrib['src'].lower()):
        validity = "mostly true"
    elif validitytext is not None and validitytext.lower() == 'unproven':
        validity = "unproven"
    else:
        error(url, "Verdict was neither True nor False nor Mixed")
        return
    savetodb(url, quotetext, validity)


while not done:
    print("========== SCRAPING PAGE " + str(pageNumber) + " ==========")
    page = requests.get("http://snopes.com/category/facts/", params={'page': pageNumber})
    pageNumber += 1
    if page.status_code != 200 or pageNumber >= 100:
        print("Got status " + str(page.status_code) + " on page " + str(pageNumber))
        done = True
        break

    htmlTree = html.fromstring(page.content)
    links = htmlTree.cssselect("div.posts-page > ul.post-list > li > div.right-side > h4 > a")

    for link in links:
        url = "http://snopes.com" + link.attrib['href']
        scrapepage(url)
        # print(url)



