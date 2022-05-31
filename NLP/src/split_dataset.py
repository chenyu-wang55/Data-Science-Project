import pickle
from sklearn.model_selection import train_test_split



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
DATASETS_MANUAL_LABELED = '../datasets/data_manual_labeled.csv'
DATASETS_COVID_19_DATASET_CSV_DIR = '../datasets/covid_19_dataset.csv'

EXPERIMENTS_BERT_RESULTS_DIR = '../experiments/fine_tuning_bert_results/'
EXPERIMENTS_BERT_RESULTS_DIR_TXT = '../experiments/fine_tuning_bert_results/fine_tuning_bert_results.txt'
EXPERIMENTS_BERT_RESULTS_TEST_PREDICTIONS_DIR = '../experiments/fine_tuning_bert_results/test_predictions.p'
EXPERIMENTS_BERT_RESULTS_COVID_19_PREDICTIONS_DIR = '../experiments/fine_tuning_bert_results/covid_19_predictions.p'


def main():
    # # specify GPU
    # device = torch.device("cuda")

    data = pickle.load(open(DATASETS_TRAINING_DATASET_DIR, 'rb'))
    data_covid_19 = pickle.load(open(DATASETS_COVID_19_DATASET_DIR, 'rb'))

    X_train = data[['text', 'Publish date']]
    y_train = data[['depression']]

    X_train, X_test_0, y_train, y_test_0 = train_test_split(X_train, y_train, test_size=0.30, stratify=y_train)
    X_test, X_val, y_test, y_val = train_test_split(X_test_0, y_test_0, test_size=0.50, stratify=y_test_0)

    print('Shapes of X_train, y_train, X_val, y_val, X_test, y_test:')
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    pickle.dump(X_train, open(DATASETS_X_TRAIN_DIR, 'wb'))
    pickle.dump(y_train, open(DATASETS_Y_TRAIN_DIR, 'wb'))
    pickle.dump(X_val, open(DATASETS_X_VAL_DIR, 'wb'))
    pickle.dump(y_val, open(DATASETS_Y_VAL_DIR, 'wb'))
    pickle.dump(X_test, open(DATASETS_X_TEST_DIR, 'wb'))
    pickle.dump(y_test, open(DATASETS_Y_TEST_DIR, 'wb'))

    X_train = pickle.load(open(DATASETS_X_TRAIN_DIR, 'rb'))
    y_train = pickle.load(open(DATASETS_Y_TRAIN_DIR, 'rb'))
    X_val = pickle.load(open(DATASETS_X_VAL_DIR, 'rb'))
    y_val = pickle.load(open(DATASETS_Y_VAL_DIR, 'rb'))
    X_test = pickle.load(open(DATASETS_X_TEST_DIR, 'rb'))
    y_test = pickle.load(open(DATASETS_Y_TEST_DIR, 'rb'))

    print('Shapes of X_train, y_train, X_val, y_val, X_test, y_test:')
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    X_test.to_csv(DATASETS_X_TEST_CSV_DIR, encoding='utf-8')
    y_test.to_csv(DATASETS_Y_TEST_CSV_DIR, encoding='utf-8')


if __name__ == '__main__':
    main()
