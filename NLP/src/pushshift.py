import pandas as pd
import requests #Pushshift accesses Reddit via an url so this is needed
import json #JSON manipulation
import csv #To Convert final table into a csv file to save to your machine
import time
import datetime

def getPushshiftData(after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?&size=2000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text, strict=False)
    return data['data']

def collect_subData(subm):
    subData = list() #list to store data points
    title = subm['title']
    #url = subm['url']
    try:
        flair = subm['link_flair_text']
    except KeyError:
        flair = "NaN"
    try:
        # returns the body of the posts
        body = subm['selftext']
    except KeyError:
        body = ''
    #author = subm['author']
    subId = subm['id']
    #score = subm['score']
    created = datetime.datetime.fromtimestamp(subm['created_utc']) #1520561700.0
    #numComms = subm['num_comments']
    #permalink = subm['permalink']

    subData.append((subId,title,body,created,flair))
    subStats[subId] = subData

def update_subFile():
    upload_count = 0
    location = "../dataset/"
    print("input filename of submission file, please add .csv")
    filename = input()
    file = location + filename
    with open(file, 'w', newline='', encoding='utf-8') as file:
        a = csv.writer(file, delimiter=',')
        headers = ["Post ID","Title","Body","Publish Date","Flair"]
        a.writerow(headers)
        for sub in subStats:
            a.writerow(subStats[sub][0])
            upload_count+=1

        print(str(upload_count) + " submissions have been uploaded into a csv file")
#global dictionary to hold 'subData'
subStats = {}
#tracks no. of submissions
subCount = 0
#Subreddit to query
sub='depression'
# Unix timestamp of date to crawl from.
before = "1643734800" #Feb 1, 2020
after = "1580576400" #Feb 1, 2022

data = getPushshiftData(after, before, sub)

while len(data) > 0:
    for submission in data:
        collect_subData(submission)
        subCount+=1
    # Calls getPushshiftData() with the created date of the last submission
    print(len(data))
    print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
    after = data[-1]['created_utc']
    data = getPushshiftData(after, before, sub)

print(len(data))

update_subFile()