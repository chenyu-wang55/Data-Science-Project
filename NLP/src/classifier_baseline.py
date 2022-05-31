
import nltk
#nltk.download('omw-1.4')
import pandas as pd
from nltk.tokenize import word_tokenize
from utilities import utils, processing_utils
import pickle
import pytorch_transformers as ppb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay, classification_report, roc_curve, auc, plot_roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from gensim.models import Doc2Vec   # gensim version=3.6.0
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import requests
import tarfile
from sklearn.decomposition import PCA

# path to dataset files.
DATASETS_DIR = '../datasets/'
DATASETS_MED_DIR = '../datasets/Meditation.csv'
DATASETS_LPT_DIR = '../datasets/LifeProTips.csv'
DATASETS_FSP_DIR = '../datasets/Friendship.csv'
DATASETS_DEPRESSION_DIR = '../datasets/Depression.csv'
DATASETS_TRAINING_DATASET_DIR = '../datasets/training_dataset.p'
DATASETS_COVID_DATASET_DIR = '../datasets/the-reddit-covid-dataset-comments.csv.zip'
DATASETS_MANUAL_LABELED_CSV = '../datasets/data_manual_labeled.csv'

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

# directories to experiment results.
EXPERIMENTS_DIR = '../experiments'
EXPERIMENTS_SIMPLE_CLASSIFIER_DIR = '../experiments/simple_classifier_results'

EXPERIMENTS_NB_CLASSIFIER_TFIDF_TXT_DIR = '../experiments/simple_classifier_results/nb_classifier_tfidf_result.txt'
EXPERIMENTS_DT_CLASSIFIER_TFIDF_TXT_DIR = '../experiments/simple_classifier_results/dt_classifier_tfidf_result.txt'
EXPERIMENTS_NB_CLASSIFIER_TFIDF_MODEL_DIR = '../experiments/simple_classifier_results/nb_classifier_tfidf_model.p'
EXPERIMENTS_DT_CLASSIFIER_TFIDF_MODEL_DIR = '../experiments/simple_classifier_results/dt_classifier_tfidf_model.p'

EXPERIMENTS_NB_CLASSIFIER_D2V_TXT_DIR = '../experiments/simple_classifier_results/nb_classifier_d2v_result.txt'
EXPERIMENTS_DT_CLASSIFIER_D2V_TXT_DIR = '../experiments/simple_classifier_results/dt_classifier_d2v_result.txt'
EXPERIMENTS_NB_CLASSIFIER_MODEL_D2V_DIR = '../experiments/simple_classifier_results/nb_classifier_d2v_model.p'
EXPERIMENTS_DT_CLASSIFIER_MODEL_D2V_DIR = '../experiments/simple_classifier_results/dt_classifier_d2v_model.p'


# path to pre_trained_model
PRETRAINED_MODEL_DIR = '../pre_trained_model'
ZIP_DIR = '../pre_trained_model/enwiki_dbow-20220306T033226Z-001.zip'
PRETRAINED_MODEL_TRAIN_D2V_DIR = '../pre_trained_model/train_doc2vec.p'
PRETRAINED_MODEL_TEST_D2V_DIR = '../pre_trained_model/test_doc2vec.p'
PRETRAINED_MODEL_TEST_100_D2V_DIR = '../pre_trained_model/test_100_doc2vec.p'
PRETRAINED_MODEL_X_TFIDF_DIR = '../pre_trained_model/x_tfidf.p'
PRETRAINED_MODEL_TRAIN_TFIDF_DIR = '../pre_trained_model/train_tfidf.p'
PRETRAINED_MODEL_TEST_TFIDF_DIR = '../pre_trained_model/test_tfidf.p'
PRETRAINED_MODEL_TEST_100_TFIDF_DIR = '../pre_trained_model/test_100_tfidf.p'



def get_zip(url):
    response = requests.get(url)
    tarPath = ZIP_DIR
    with open(tarPath, 'wb') as file:
        file.write(response.content)
        file.flush()
    tar = tarfile.open(tarPath)
    tar.extractall(PRETRAINED_MODEL_DIR)
    tar.close()
    os.remove(tarPath)
    print('finished download')

def clean_text(text, flg_stemm=False, flg_lemm=True):
    text = text.values.tolist()
    lst_stopwords = nltk.corpus.stopwords.words("english")
    ps = nltk.stem.porter.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    text_cleaned = []
    for word in text:
        tokenized_word = nltk.word_tokenize(word)
        # remove stopwords, numbers, word length less than 1
        lst_text = [w for w in tokenized_word if
                    (len(w) > 2 and w not in lst_stopwords)]

        # Stemming (remove -ing, -ly, ...)
        if flg_stemm == True:
            lst_text = [ps.stem(word) for word in lst_text]

        # Lemmatisation (convert the word into root word)
        if flg_lemm == True:
            lst_text = [lem.lemmatize(word) for word in lst_text]

        lst_str = ' '.join(lst_text)
        #print(lst_str)
        text_cleaned.append(lst_str)
        #print(text_cleaned)

    return text_cleaned

def get_tfidf(text):
    corpus = clean_text(text)
    vectorizer = TfidfVectorizer()
    text_tfidf = vectorizer.fit_transform(corpus)
    text_tfidf = text_tfidf.toarray()
    pca = PCA(n_components=300)
    text_tfidf = pca.fit_transform(text_tfidf)

    return text_tfidf

def get_doc2vec(text, need_clean=True):
    '''

    :return:
    '''
    if need_clean == True:
        doc = clean_text(text)
    else:
        doc = text.values.tolist()
    model = '../pre_trained_model/enwiki_dbow/doc2vec.bin'
    # inference hyper-parameters
    start_alpha = 0.01
    infer_epoch = 100

    # load model
    m = Doc2Vec.load(model)
    text_d2v = []
    for d in doc:
        text_d2v.append(m.infer_vector(d, alpha=start_alpha, steps=infer_epoch))

    text_d2v = np.array(text_d2v)

    return text_d2v

def NB_classifier(X_train, y_train):
    '''

    :return:
    '''
    param_grid_nb = {
        'var_smoothing': np.logspace(0, -2, num=100)}
    nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, cv=5, verbose=1,  n_jobs=-1)
    nbModel_grid.fit(X_train, y_train)
    model_nb = nbModel_grid.best_estimator_

    return model_nb

def dt_classifier(X_train, y_train):
    '''

    :return:
    '''
    dt = DecisionTreeClassifier(random_state=42)
    params = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_leaf': [10, 20, 30, 40],
        'criterion': ["entropy"]
    }
    grid_search_dt = GridSearchCV(estimator=dt,
                                  param_grid=params,
                                    cv=5, n_jobs=-1, verbose=1, scoring="accuracy")

    grid_search_dt.fit(X_train, y_train)
    best_model_dt = grid_search_dt.best_estimator_

    return best_model_dt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

def classifier_result(model, X_test, y_test, str, dir_txt):
    '''

    :return:
    '''
    nb_pred = model.predict(X_test)
    utils.append_to_text_file_and_print_line(dir_txt, "Classification Report of " + str + " is:\n" + classification_report(y_test, nb_pred))


    print("Classification Report is:\n", classification_report(y_test, nb_pred))
    print("\n Confusion Matrix:\n")

    plt.figure()
    cm = confusion_matrix(y_test, nb_pred)

    plot_confusion_matrix(cm, classes=[0, 1], normalize=True,
                          title='Confusion Matrix')
    path = EXPERIMENTS_SIMPLE_CLASSIFIER_DIR + '/confusion_matrix ' + ' ' + str
    plt.savefig(path)




def main():
    X_train = pickle.load(open(DATASETS_X_TRAIN_DIR, 'rb'))
    y_train = pickle.load(open(DATASETS_Y_TRAIN_DIR, 'rb'))
    X_test = pickle.load(open(DATASETS_X_TEST_DIR, 'rb'))
    y_test = pickle.load(open(DATASETS_Y_TEST_DIR, 'rb'))
    test_100 = pd.read_csv(DATASETS_MANUAL_LABELED_CSV)

    #training and test set
    X_train = X_train['text']
    X_test = X_test['text']
    X_test_100 = test_100['text']

    # covert the text column in X_train and X_test
    if not os.path.exists(PRETRAINED_MODEL_X_TFIDF_DIR):
         X = pd.concat([X_train, X_test, X_test_100])
         X_tfidf = get_tfidf(X)
         pickle.dump(X_tfidf, open(PRETRAINED_MODEL_X_TFIDF_DIR, 'wb'))

    X_tfidf = pickle.load(open(PRETRAINED_MODEL_X_TFIDF_DIR, 'rb'))

    X_train_tfidf = X_tfidf[:14000]
    X_test_tfidf = X_tfidf[14000:17000]
    X_test_100_tfidf = X_tfidf[17000:]
    pickle.dump(X_train_tfidf, open(PRETRAINED_MODEL_TRAIN_TFIDF_DIR, 'wb'))
    pickle.dump(X_test_tfidf, open(PRETRAINED_MODEL_TEST_TFIDF_DIR, 'wb'))
    pickle.dump(X_test_100_tfidf, open(PRETRAINED_MODEL_TEST_100_TFIDF_DIR, 'wb'))

    print(X_train_tfidf.shape)
    print(X_test_tfidf.shape)
    print(X_test_100_tfidf.shape)

    # Encode the string categorical value
    le = LabelEncoder()
    y_train = y_train['depression']
    y_test = y_test['depression']
    y_test_100 = test_100['depression']
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    y_test_100 = le.fit_transform(y_test_100)

    if not os.path.exists('../pre_trained_model/enwiki_dbow'):
        get_zip('https://cloudstor.aarnet.edu.au/plus/s/hpfhUC72NnKDxXw/download')

    #covert the text column in X_train
    if not os.path.exists(PRETRAINED_MODEL_TRAIN_D2V_DIR):
        train_doc2vec = get_doc2vec(X_train)
        pickle.dump(train_doc2vec, open(PRETRAINED_MODEL_TRAIN_D2V_DIR, 'wb'))

    train_doc2vec = pickle.load(open(PRETRAINED_MODEL_TRAIN_D2V_DIR, 'rb'))

    print(train_doc2vec.shape)
    
    #covert the text column in X_test
    if not os.path.exists(PRETRAINED_MODEL_TEST_D2V_DIR):
        test_doc2vec = get_doc2vec(X_test)
        pickle.dump(test_doc2vec, open(PRETRAINED_MODEL_TEST_D2V_DIR, 'wb'))

    test_doc2vec = pickle.load(open(PRETRAINED_MODEL_TEST_D2V_DIR, 'rb'))

    print(test_doc2vec.shape)
    
    # covert the body column in X_test_100
    if not os.path.exists(PRETRAINED_MODEL_TEST_100_D2V_DIR):
        X_test_100_doc2vec = get_doc2vec(X_test_100)
        pickle.dump(X_test_100_doc2vec, open(PRETRAINED_MODEL_TEST_100_D2V_DIR, 'wb'))

    X_test_100_doc2vec = pickle.load(open(PRETRAINED_MODEL_TEST_100_D2V_DIR, 'rb'))
    print(X_test_100_doc2vec.shape)


    #NB classifer_tfidf
    if not os.path.exists(EXPERIMENTS_NB_CLASSIFIER_TFIDF_MODEL_DIR):
        depression_nb_tfidf = NB_classifier(X_train_tfidf, y_train)

        pickle.dump(depression_nb_tfidf, open(EXPERIMENTS_NB_CLASSIFIER_TFIDF_MODEL_DIR, 'wb'))

    depression_nb_tfidf = pickle.load(open(EXPERIMENTS_NB_CLASSIFIER_TFIDF_MODEL_DIR, 'rb'))

    # NB classifer_d2v
    if not os.path.exists(EXPERIMENTS_NB_CLASSIFIER_MODEL_D2V_DIR):
        depression_nb_d2v = NB_classifier(train_doc2vec, y_train)

        pickle.dump(depression_nb_d2v, open(EXPERIMENTS_NB_CLASSIFIER_MODEL_D2V_DIR, 'wb'))

    depression_nb_d2v = pickle.load(open(EXPERIMENTS_NB_CLASSIFIER_MODEL_D2V_DIR, 'rb'))


    #DT classifer_tfidf
    if not os.path.exists(EXPERIMENTS_DT_CLASSIFIER_TFIDF_MODEL_DIR):
        depression_dt_tfidf = dt_classifier(X_train_tfidf, y_train)

        pickle.dump(depression_dt_tfidf, open(EXPERIMENTS_DT_CLASSIFIER_TFIDF_MODEL_DIR, 'wb'))

    depression_dt_tfidf = pickle.load(open(EXPERIMENTS_DT_CLASSIFIER_TFIDF_MODEL_DIR, 'rb'))

    # DT classifer_d2v
    if not os.path.exists(EXPERIMENTS_DT_CLASSIFIER_MODEL_D2V_DIR):
        depression_dt_d2v = dt_classifier(train_doc2vec, y_train)
        pickle.dump(depression_dt_d2v, open(EXPERIMENTS_DT_CLASSIFIER_MODEL_D2V_DIR, 'wb'))

    depression_dt_d2v = pickle.load(open(EXPERIMENTS_DT_CLASSIFIER_MODEL_D2V_DIR, 'rb'))

    #result nb & dt with tfidf
    classifier_result(depression_nb_tfidf, X_test_tfidf, y_test, '30% test set on_nb_tfidf', EXPERIMENTS_NB_CLASSIFIER_TFIDF_TXT_DIR)
    classifier_result(depression_nb_tfidf, X_test_100_tfidf, y_test_100, '100 sample on_nb_tfidf', EXPERIMENTS_NB_CLASSIFIER_TFIDF_TXT_DIR)

    classifier_result(depression_dt_tfidf, X_test_tfidf, y_test, '30% test set on_dt_tfidf',
                      EXPERIMENTS_DT_CLASSIFIER_TFIDF_TXT_DIR)
    classifier_result(depression_dt_tfidf, X_test_100_tfidf, y_test_100, '100 sample on_dt_tfidf',
                      EXPERIMENTS_DT_CLASSIFIER_TFIDF_TXT_DIR)

    # result nb & dt with d2v
    classifier_result(depression_nb_d2v, test_doc2vec, y_test, '30% test set on_nb_d2v',
                      EXPERIMENTS_NB_CLASSIFIER_D2V_TXT_DIR)
    classifier_result(depression_nb_d2v, X_test_100_doc2vec, y_test_100, '100 sample on_nb_d2v',
                      EXPERIMENTS_NB_CLASSIFIER_D2V_TXT_DIR)

    classifier_result(depression_dt_d2v, test_doc2vec, y_test, '30% test set on_dt_d2v',
                      EXPERIMENTS_DT_CLASSIFIER_D2V_TXT_DIR)
    classifier_result(depression_dt_d2v, X_test_100_doc2vec, y_test_100, '100 sample on_dt_d2v',
                      EXPERIMENTS_DT_CLASSIFIER_D2V_TXT_DIR)


if __name__ == '__main__':
    main()
