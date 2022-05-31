from nltk.tokenize import word_tokenize
import nltk
from utilities import utils, processing_utils
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import sklearn
from sklearn.metrics import RocCurveDisplay, classification_report, roc_curve, auc, plot_roc_curve
from sklearn.model_selection import train_test_split
import transformers
from transformers import AutoModel, BertTokenizer, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from Bert import BERT_Arch
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
import time
from utilities import utils


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


def get_seq_mask(texts, tokenizer, padding_len):
    # tokenize and encode sequences in the test set
    tokens_train_unlabeled = tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=padding_len,
        padding=True,
        truncation=True
    )
    seq = torch.tensor(tokens_train_unlabeled['input_ids'])
    mask = torch.tensor(tokens_train_unlabeled['attention_mask'])
    return seq, mask


def get_dataloader_random_sampler(padding_len, tokenizer, X, y):
    # tokenize and encode sequences in the training set
    tokens = tokenizer.batch_encode_plus(
        X.tolist(),
        max_length=padding_len,
        padding=True,
        truncation=True
    )

    ## convert lists to tensors
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    y_tensor = torch.tensor(y.tolist())
    # define a batch size
    batch_size = 64
    # wrap tensors
    data_tensor = TensorDataset(seq, mask, y_tensor)
    # sampler for sampling the data during training
    sampler = RandomSampler(data_tensor)
    # dataLoader for train set
    dataloader = DataLoader(data_tensor, sampler=sampler, batch_size=batch_size)
    return dataloader


def get_dataloader_sequential_sampler(padding_len, tokenizer, X, y):
    # tokenize and encode sequences in the training set
    tokens = tokenizer.batch_encode_plus(
        X.tolist(),
        max_length=padding_len,
        padding=True,
        truncation=True
    )

    ## convert lists to tensors
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    y_tensor = torch.tensor(y.tolist())
    # define a batch size
    batch_size = 64
    # wrap tensors
    data_tensor = TensorDataset(seq, mask, y_tensor)
    # sampler for sampling the data during training
    sampler = SequentialSampler(data_tensor)
    # dataLoader for train set
    dataloader = DataLoader(data_tensor, sampler=sampler, batch_size=batch_size)
    return dataloader


def get_unlabeled_data_dataloader(padding_len, tokenizer, X):
    seq, mask = get_seq_mask(X, tokenizer, padding_len)

    # define a batch size
    batch_size = 64
    # wrap tensors
    data_tensor = TensorDataset(seq, mask)
    # sampler for sampling the data during training
    sampler = SequentialSampler(data_tensor)
    # dataLoader for train set
    dataloader = DataLoader(data_tensor, sampler=sampler, batch_size=batch_size)
    return dataloader


def evaluate(model, val_dataloader, cross_entropy):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 20 batches.
        if step % 20 == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # # push the batch to gpu
        # batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    print('\nFinished evaluating.')
    return avg_loss, total_preds


def evaluate_unlabeled(model, dataloader):
    """
    Function for labelling the unlabeled dataset by the model

    :param model:
    :param dataloader:
    :return:
    """
    print("\nEvaluating unlabeled...")

    # deactivate dropout layers
    model.eval()

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(dataloader):

        # Progress update every 20 batches.
        if step % 20 == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        # # push the batch to gpu
        # batch = [t.to(device) for t in batch]

        sent_id, mask = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    print('\nFinished predicting unlabeled data.')
    return total_preds


def train_one_epoch(model, train_dataloader, loss_fn, optimizer):
    """
    Train for 1 epoch.
    :param model:
    :param train_dataloader:
    :param loss_fn:
    :param optimizer:
    :return:
    """
    print("\nTraining...")

    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        # progress update after every 20 batches.
        if (step+1) % 20 == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step+1, len(train_dataloader)))

        # # push the batch to gpu
        #batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = loss_fn(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


def training_and_evaluating(model, loss_fn, optimizer, epochs, padding_len, tokenizer, X_train, X_val, y_train, y_val):
    X_train_0 = np.array(X_train)
    X_val_0 = np.array(X_val)
    y_train_0 = np.array(y_train)
    y_val_0 = np.array(y_val)

    train_dataloader = get_dataloader_random_sampler(padding_len, tokenizer, X_train_0, y_train_0)
    val_dataloader = get_dataloader_sequential_sampler(padding_len, tokenizer, X_val_0, y_val_0)

    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    t0 = time.time()

    # for each epoch
    for epoch in range(epochs):

        print('\nEpoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train_one_epoch(model, train_dataloader, loss_fn, optimizer)
        # train_loss = 0
        print('\nFinished training on training data for this epoch.')

        # evaluate model
        valid_loss, preds_labeled_probs = evaluate(model, val_dataloader, loss_fn)
        y_preds_labels = np.argmax(preds_labeled_probs, axis=1)
        print('\nValidation accuracy for epoch {} is {}'.format(epoch+1, float(sum(np.equal(y_val_0, y_preds_labels))) / len(y_preds_labels)))

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(EXPERIMENTS_BERT_RESULTS_DIR, 'saved_weights_best.pt'))
        torch.save(model.state_dict(), os.path.join(EXPERIMENTS_BERT_RESULTS_DIR, 'saved_weights_last.pt'))

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print(f'\nTraining Loss: {train_loss:.3f}')
        # print(f'Validation Loss: {valid_loss:.3f}')

        utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, '\nIn Epoch {}, Training Loss is {:.3f} and Validation Loss is {:.3f}'.format(epoch + 1, train_loss, valid_loss))
        utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Finished Training')
        utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Total time used for training and evaluating is {}'.format(time.time() - t0))
    return train_losses, valid_losses


def main():
    st = time.time()
    # # specify GPU
    # device = torch.device("cuda")

    X_train = pickle.load(open(DATASETS_X_TRAIN_DIR, 'rb'))
    y_train = pickle.load(open(DATASETS_Y_TRAIN_DIR, 'rb'))
    X_val = pickle.load(open(DATASETS_X_VAL_DIR, 'rb'))
    y_val = pickle.load(open(DATASETS_Y_VAL_DIR, 'rb'))
    X_test = pickle.load(open(DATASETS_X_TEST_DIR, 'rb'))
    y_test = pickle.load(open(DATASETS_Y_TEST_DIR, 'rb'))
    data_covid_19 = pickle.load(open(DATASETS_COVID_19_DATASET_DIR, 'rb'))
    data_manual_labeled = pd.read_csv(DATASETS_MANUAL_LABELED_CSV)

    X_train = X_train.replace(r'\s+', ' ', regex=True)
    X_val = X_val.replace(r'\s+', ' ', regex=True)
    X_test = X_test.replace(r'\s+', ' ', regex=True)
    data_manual_labeled = data_manual_labeled.replace(r'\s+', ' ', regex=True)

    # all the columns:
    # X_train = data[['text', 'Publish date']]
    # y_train = data[['depression']]
    # x_covid_19 = data_covid_19[['body']]

    # convert "yes", "no" to 1 and 0.
    y_train.loc[y_train['depression'] == 'no', 'depression'] = 0
    y_train.loc[y_train['depression'] == 'yes', 'depression'] = 1
    y_val.loc[y_val['depression'] == 'no', 'depression'] = 0
    y_val.loc[y_val['depression'] == 'yes', 'depression'] = 1
    y_test.loc[y_test['depression'] == 'no', 'depression'] = 0
    y_test.loc[y_test['depression'] == 'yes', 'depression'] = 1
    data_manual_labeled.loc[data_manual_labeled['depression'] == 'no', 'depression'] = 0
    data_manual_labeled.loc[data_manual_labeled['depression'] == 'yes', 'depression'] = 1

    X_train = X_train['text']
    X_val = X_val['text']
    X_test = X_test['text']
    y_train = y_train['depression']
    y_val = y_val['depression']
    y_test = y_test['depression']
    X_data_manual_labeled = data_manual_labeled['text']
    y_data_manual_labeled = data_manual_labeled['depression']
    y_data_manual_labeled = torch.tensor(y_data_manual_labeled.tolist())
    x_covid_19 = data_covid_19['body']

    print('Shapes of X_train, y_train, X_val, y_val, X_test, y_test, x_covid_19:')
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape, x_covid_19.shape)

    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # get length of all the messages in the train set
    seq_lengths = [len(i.split()) for i in X_train]
    # hist = pd.Series(seq_lengths).hist(bins=50)
    # plt.show()
    max_seq_len = max(seq_lengths)
    print('max_seq_len', max_seq_len)  # 6720 in training data
    print('avg_seq_len', np.mean(seq_lengths))

    padding_len = 100

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        X_test.tolist(),
        max_length=padding_len,
        padding=True,
        truncation=True
    )
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(y_test.tolist())

    # # freeze all the parameters
    # for param in bert.parameters():
    #     param.requires_grad = False

    # pass the pre-trained BERT to our define architecture
    model = BERT_Arch(bert)

    # # push the model to GPU
    # model = model.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    #compute the class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    print("Class Weights:", class_weights)

    # converting list of class weights to a tensor
    weights = torch.tensor(class_weights, dtype=torch.float)

    # # push to GPU
    # weights = weights.to(device)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    # number of training epochs
    epochs = 3

    # train_losses, valid_losses = training_and_evaluating(model, cross_entropy, optimizer, epochs, padding_len, tokenizer,
    #                                                      X_train, X_val, y_train, y_val)
    model.load_state_dict(torch.load(os.path.join(EXPERIMENTS_BERT_RESULTS_DIR, 'saved_weights_last.pt')))

    # get predictions for test data
    with torch.no_grad():
        # utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Started prediction of test data.')
        # test_data_loader = get_dataloader_sequential_sampler(padding_len, tokenizer, X_test, y_test)
        # test_loss, preds_labeled_probs = evaluate(model, test_data_loader, cross_entropy)
        # preds_labeled = np.argmax(preds_labeled_probs, axis=1)
        # print(classification_report(test_y, preds_labeled))
        # pickle.dump(preds_labeled, open(EXPERIMENTS_BERT_RESULTS_TEST_PREDICTIONS_DIR, 'wb'))
        # utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Finished prediction of test data.')
        # utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Final accuracy on labeled testing set ' + str(np.sum(np.equal(preds_labeled, np.array(test_y))) / float(len(preds_labeled))))

        utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Started prediction of unlabeled test data.')
        unlabeled_dataloader = get_unlabeled_data_dataloader(padding_len, tokenizer, x_covid_19)
        preds_unlabeled_probs = evaluate_unlabeled(model, unlabeled_dataloader)
        print('preds_unlabeled_probs ', preds_unlabeled_probs[:10])
        preds_unlabeled = np.argmax(preds_unlabeled_probs, axis=1)
        print('preds_unlabeled_probs ', preds_unlabeled[:10])
        pickle.dump(preds_unlabeled, open(EXPERIMENTS_BERT_RESULTS_COVID_19_PREDICTIONS_DIR, 'wb'))
        utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Finished prediction of unlabeled test data.')
        data_covid_19['predictions'] = preds_unlabeled.tolist()
        data_covid_19.to_csv(EXPERIMENTS_COVID_19_DATASET_PREDICTED_CSV_DIR, encoding='utf-8')

        # utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Started prediction of manually labeled test data.')
        # test_data_loader = get_dataloader_sequential_sampler(padding_len, tokenizer, X_data_manual_labeled, y_data_manual_labeled)
        # test_loss, preds_labeled_probs = evaluate(model, test_data_loader, cross_entropy)
        # preds_labeled = np.argmax(preds_labeled_probs, axis=1)
        # print(classification_report(y_data_manual_labeled, preds_labeled))
        # pickle.dump(preds_labeled, open(EXPERIMENTS_BERT_RESULTS_TEST_100_PREDICTIONS_DIR, 'wb'))
        # utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Finished prediction of manually labeled test data.')
        # utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Final accuracy on labeled testing set ' + str(np.sum(np.equal(preds_labeled, np.array(y_data_manual_labeled))) / float(len(preds_labeled))))
        # print(y_data_manual_labeled.tolist())
        # print(preds_labeled.tolist())
        # data_manual_labeled['predictions'] = preds_labeled.tolist()
        # data_manual_labeled.to_csv(EXPERIMENTS_MANUAL_LABELED_PREDICTED_CSV, encoding='utf-8')
    utils.append_to_text_file_and_print_line(EXPERIMENTS_BERT_RESULTS_DIR_TXT, 'Total time used for the program is {}'.format(time.time() - st))


if __name__ == '__main__':
    main()
