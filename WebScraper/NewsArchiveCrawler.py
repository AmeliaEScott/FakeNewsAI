import csv

with open("NewsAggregatorDataset/newsCorpora.csv") as csvFile:
    reader = csv.reader(csvFile, dialect=csv.excel_tab)
    for row in reader:
        # print(repr(row))
        id = row[0]
        title = row[1]
        url = row[2]
        category = row[4]  # b = business, t = science and technology, e = entertainment, m = health
        timestamp = row[7]

        stuff = {
            'id': id,
            'title': title,
            'url': url,
            'category': category,
            'timestamp': timestamp
        }

        print(repr(stuff))
