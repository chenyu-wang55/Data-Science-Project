import pandas as pd
import numpy as np
import os
import pickle
import nltk
import re

np.random.seed(seed=2)

DATASETS_DIR = '../datasets/'
DATASETS_MED_DIR = '../datasets/Meditation.csv'
DATASETS_LPT_DIR = '../datasets/LifeProTips.csv'
DATASETS_FSP_DIR = '../datasets/Friendship.csv'
DATASETS_DEPRESSION_DIR = '../datasets/Depression.csv'
DATASETS_TRAINING_DATASET_DIR = '../datasets/training_dataset.p'
DATASETS_COVID_DATASET_DIR = '../datasets/the-reddit-covid-dataset-comments.csv.zip'
DATASETS_COVID_19_DATASET_DIR = '../datasets/covid_19_dataset.p'
DATASETS_COVID_19_DATASET_CSV_DIR = '../datasets/covid_19_dataset.csv'
#DATASET_P_DIR2 = '../dataset/LifeProTips.p'

def choose_records(df):
    #df["Content"] = df["Title"] + " " + df["Body"]
    #df["Content"] = df["Body"]
    df = df.drop(columns=['Post_iD', 'Title'])
    df = df[df.Body.apply(lambda x: len(str(x).split()) >= 40)]

    return df

def no_depression_data(df, keywords):
    text = df['Body'].values.tolist()
    dates = df['Publish_date'].values.tolist()
    text_cleaned = []
    time = []
    count = 0
    #print(len(text))
    for record, date in zip(text, dates):
        letters_only = re.sub("[^A-Za-z0-9.]", " ", str(record).lower())
        letters_only = re.sub('\n', '', letters_only)
        n = 0
        for keyword in keywords:
            if keyword in letters_only:
                count += 1
                break
            else:
                n += 1
                if n == len(keywords):
                    text_cleaned.append(letters_only)
                    time.append(date)
                else:
                    continue

    label = ['no']*len(text_cleaned)
    #print(count)
    #print(len(label))

    #print(len(text_cleaned))
    #print(len(time))
    #print(text_cleaned[:3])
    c = {'text': text_cleaned, 'Publish date': time, 'depression': label}
    new_df = pd.DataFrame(c)
    #df['text'] = text_cleaned
    #print(new_df.info)

    return new_df

def depression_data(df, keywords):
    text = df['Body'].values.tolist()
    dates = df['Publish_date'].values.tolist()
    text_cleaned = []
    time = []
    count = 0
    #print(len(text))
    for record, date in zip(text, dates):
        letters_only = re.sub("[^A-Za-z0-9.]", " ", str(record).lower())
        letters_only = re.sub('\n', '', letters_only)
        for keyword in keywords:
            if keyword in letters_only:
                text_cleaned.append(letters_only)
                time.append(date)
                count += 1
                break
            else:
                continue

    label = ['yes']*len(text_cleaned)
    #print(count)
    #print(len(label)

    #print(len(text_cleaned))
    #print(text_cleaned[:3])

    c = {'text': text_cleaned, 'Publish date': time, 'depression': label}
    new_df = pd.DataFrame(c)
    new_df = new_df.sample(n=10000, random_state=1)
    #df['text'] = text_cleaned
    #print(new_df.info)

    return new_df


meditation_dataset = pd.read_csv(DATASETS_MED_DIR)
lifeProTips_dataset = pd.read_csv(DATASETS_LPT_DIR)
friendship_dataset = pd.read_csv(DATASETS_FSP_DIR)
depression_dataset = pd.read_csv(DATASETS_DEPRESSION_DIR)
#training_set =  pd.concat([])

depression_keyword = ['alone', 'break', 'blame', 'depressed', 'deserve better',
                      'deserve unhappy', 'die', 'escape', 'distraction', 'nobody',
                      'feel alone', 'feel depressed', 'felt pain', 'fuck don', 'hate',
                      'hurt', 'loneliness', 'mine', 'myself', 'reject love', 'safe',
                      'shit', 'sucks', 'no job', 'painful', 'pressure', 'too worried',
                      'unsuccessful', 'ugly', 'uncomfortable', 'winter', 'worry',
                      'worth', 'wrong life']

meditation_dataset_no = no_depression_data(choose_records(meditation_dataset), depression_keyword)
lifeProTips_dataset_no = no_depression_data(choose_records(lifeProTips_dataset), depression_keyword)
friendship_dataset_no = no_depression_data(choose_records(friendship_dataset), depression_keyword)
no_depression_dataset = pd.concat([meditation_dataset_no,lifeProTips_dataset_no, friendship_dataset_no])
no_depression_dataset = no_depression_dataset.sample(n=10000, random_state=1)
print(no_depression_dataset.info)

depression_dataset_yes = depression_data(choose_records(depression_dataset), depression_keyword)
print(depression_dataset_yes.info)

training_dateset = pd.concat([no_depression_dataset,depression_dataset_yes])
pickle.dump(training_dateset, open(DATASETS_TRAINING_DATASET_DIR, 'wb'))
print(training_dateset.info)
