
import pandas as pd
import requests #Pushshift accesses Reddit via an url so this is needed
import json #JSON manipulation
import csv #To Convert final table into a csv file to save to your machine
import time
import datetime
import os
import pickle

DATASETS_DIR = '../datasets/'
DATASETS_MED_DIR = '../datasets/Meditation.csv'
DATASETS_LPT_DIR = '../datasets/LifeProTips.csv'
DATASETS_FSP_DIR = '../datasets/Friendship.csv'
DATASETS_DEPRESSION_DIR = '../datasets/Depression.csv'


def getPushshiftData(after, before, sub):
    #Build URL
    url = 'https://api.pushshift.io/reddit/search/submission/?&size=2000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    #Print URL to show user
    print(url)
    #Request URL
    r = requests.get(url)
    #Load JSON data from webpage into data variable
    data = json.loads(r.text)
    #return the data element which contains all the submissions data
    # print("No Error")
    return data['data']

def collectSubData(subm):
    #subData was created at the start to hold all the data which is then added to our global subStats dictionary.
    subData = list() #list to store data points
    title = subm['title']
    sub_id = subm['id']
    #score = subm['score']
    created = datetime.datetime.fromtimestamp(subm['created_utc']) #1520561700.0
    #numComms = subm['num_comments']
    selftext = subm['selftext']
    subreddit = subm['subreddit']
    #over_18 = subm['over_18']
    #Put all data points into a tuple and append to subData
    subData.append((sub_id,title,selftext,created))
    #Create a dictionary entry of current submission data and store all data related to it
    subStats[sub_id] = subData

#Create your timestamps and queries for your search URL
#https://www.unixtimestamp.com/index.php > Use this to create your timestamps
#after = "1580576400" #Submissions after this timestamp (1577836800 = 01 Feb 20)
after = '1612155600'#Submissions after this timestamp (1577836800 = 01 Feb 21)
before = "1643734800" #Submissions before this timestamp (1607040000 = 01 Feb 22)
#Keyword(s) to look for in submissions
#sub = "depression" #Which Subreddit to search in
#sub = "Meditation" #Which Subreddit to search in
sub = "LifeProTips" #Which Subreddit to search in

#subCount tracks the no. of total submissions we collect
subCount = 0
#subStats is the dictionary where we will store our data.
subStats = {}

# We need to run this function outside the loop first to get the updated after variable
data = getPushshiftData(after, before, sub)
# Will run until all posts have been gathered i.e. When the length of data variable = 0
# from the 'after' date up until before date
while len(data) > 0: #The length of data is the number submissions (data[0], data[1] etc), once it hits zero (after and before vars are the same) end
    for submission in data:
        try:
            collectSubData(submission)
            subCount += 1
        except:
            continue
    # Calls getPushshiftData() with the created date of the last submission
    # print(len(data))
    # print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
    #update after variable to last created date of submission
    after = data[-1]['created_utc']
    # print("CHECK")
    #data has changed due to the new after variable provided by above code
    try:
        data = getPushshiftData(after, before, sub)
        print(subCount)
    except:
        if int(after) >= int(before):
            break

print(len(data))
print(str(len(subStats)) + " submissions have added to list")
#print("1st entry is:")
#print(list(subStats.values())[0][0][1] + " created: " + str(list(subStats.values())[0][0][5]))
#print("Last entry is:")
#print(list(subStats.values())[-1][0][1] + " created: " + str(list(subStats.values())[-1][0][5]))


def updateSubs_file():
    upload_count = 0

    filename = DATASETS_MED_DIR
    #filename = DATASETS_FSP_DIR
    #filename = DATASETS_LPT_DIR
    #filename = DATASETS_DEPRESSION_DIR

    file = filename
    with open(file, 'w', newline='', encoding='utf-8') as file:
        a = csv.writer(file, delimiter=',')
        headers = ["Post_iD", "Title", "Body", "Publish_date"]
        a.writerow(headers)
        for sub in subStats:
            a.writerow(subStats[sub][0])
            upload_count += 1

        print(str(upload_count) + " submissions have been uploaded")

updateSubs_file()


#df = pd.read_csv(DATASET_CSV_DIR1)
#pickle.dump(df, open(os.path.join(DATASET_DIR, DATASET_P_DIR1), 'wb'))

