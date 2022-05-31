import pandas as pd
import numpy as np
import os
import pickle
import nltk
import re
import time

DATASETS_DIR = '../datasets/'
DATASETS_MED_DIR = '../datasets/Meditation.csv'
DATASETS_LPT_DIR = '../datasets/LifeProTips.csv'
DATASETS_FSP_DIR = '../datasets/Friendship.csv'
DATASETS_DEPRESSION_DIR = '../datasets/Depression.csv'
DATASETS_TRAINING_DATASET_DIR = '../datasets/training_dataset.p'
DATASETS_COVID_DATASET_DIR = '../datasets/the-reddit-covid-dataset-comments.csv.zip'

DATASETS_X_TRAIN_DIR = '../datasets/x_train.p'
DATASETS_Y_TRAIN_DIR = '../datasets/y_train.p'
DATASETS_X_VAL_DIR = '../datasets/x_val.p'
DATASETS_Y_VAL_DIR = '../datasets/y_val.p'
DATASETS_X_TEST_DIR = '../datasets/x_test.p'
DATASETS_Y_TEST_DIR = '../datasets/y_test.p'
DATASETS_COVID_19_DATASET_DIR = '../datasets/covid_19_dataset.p'

DATASETS_X_TEST_CSV_DIR = '../datasets/x_test.csv'
DATASETS_Y_TEST_CSV_DIR = '../datasets/y_test.csv'
DATASETS_MANUAL_LABELED_CSV = '../datasets/data_manual_labeled.csv'
DATASETS_COVID_19_DATASET_CSV_DIR = '../datasets/covid_19_dataset.csv'

EXPERIMENTS_BERT_RESULTS_DIR = '../experiments/fine_tuning_bert_results/'
EXPERIMENTS_BERT_RESULTS_DIR_TXT = '../experiments/fine_tuning_bert_results/fine_tuning_bert_results.txt'
EXPERIMENTS_BERT_RESULTS_TEST_PREDICTIONS_DIR = '../experiments/fine_tuning_bert_results/test_predictions.p'
EXPERIMENTS_BERT_RESULTS_TEST_100_PREDICTIONS_DIR = '../experiments/fine_tuning_bert_results/test_100_predictions.p'
EXPERIMENTS_BERT_RESULTS_COVID_19_PREDICTIONS_DIR = '../experiments/fine_tuning_bert_results/covid_19_predictions.p'
EXPERIMENTS_MANUAL_LABELED_PREDICTED_CSV = '../experiments/fine_tuning_bert_results/data_manual_labeled_predicted.csv'
EXPERIMENTS_COVID_19_DATASET_PREDICTED_CSV_DIR = '../experiments/fine_tuning_bert_results/covid_19_dataset_predicted.csv'


def main():

    # sample_cols_to_keep = ['id', 'subreddit.id', 'subreddit.name', 'created_utc', 'body', 'sentiment', 'score']
    sample_cols_to_keep = ['created_utc', 'body', 'sentiment', 'score']

    # depression_indicative_phrases = [
    #     'alone', 'break', 'blame', 'depressed', 'deserve better', 'deserve unhappy', 'die', 'escape', 'distraction',
    #     'nobody', 'feel alone', 'feel depressed', 'felt pain', 'fuck don\'t', 'hate', 'hurt', 'loneliness', 'mine', 'myself',
    #     'reject love', 'safe', 'shit', 'sucks', 'no job', 'painful', 'pressure', 'too worried', 'unsuccessful', 'ugly',
    #     'uncomfortable', 'winter', 'worry', 'worth', 'wrong life'
    # ]

    # First setup dataframe iterator, ‘usecols’ parameter filters the columns, and 'chunksize' sets the number of rows per chunk in the csv. (you can change these parameters as you wish)
    df_iter = pd.read_csv(DATASETS_COVID_DATASET_DIR, compression='zip', chunksize=30000, usecols=sample_cols_to_keep)
    f = lambda t: len(t.split())
    f_vectorized = np.vectorize(f)

    length = 40

    # this list will store the filtered dataframes for later concatenation
    df_list = []

    i = 0
    st = time.time()
    # Iterate over the file based on the criteria and append to the list
    for df in df_iter:
        # tmp_df = df.rename(columns={col: col.lower() for col in df.columns})
        tmp_df = df[df.body.apply(lambda x:  len(str(x).split()) >= length)]
        tmp_df = tmp_df.replace('[^A-Za-z0-9.]', ' ', regex=True).replace(r'\n', ' ', regex=True).replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\s+', ' ', regex=True)

        i += 1
        if i % 10 == 0:
            print('i', i)
        if len(tmp_df) > 0:
            df_list += [tmp_df.copy()]
            # tmp_df_body = np.array(tmp_df.body)
            # f1 = f_vectorized(tmp_df_body)
            # assert all(i >= length for i in f1)
    print('Total time used is', time.time() - st)
    print('i is', i)

    # And finally combine filtered df_lst into the final laeger output say 'df_final' dataframe
    df_final = pd.concat(df_list)
    print(len(df_final))
    df_final = df_final.sample(n=20000, random_state=1)
    pickle.dump(df_final, open(DATASETS_COVID_19_DATASET_DIR, 'wb'))
    print('Finished dumping COVID-19 testing dataset.')


if __name__ == '__main__':
    main()
    testing_dataset = pickle.load(open(DATASETS_COVID_19_DATASET_DIR, 'rb'))
    print(len(testing_dataset))
    num = 20
    result = testing_dataset.head(num)
    print("First {} rows of the DataFrame:".format(num))
    print(result)
    print(result['body'])
    testing_dataset.to_csv(DATASETS_COVID_19_DATASET_CSV_DIR, encoding='utf-8')
