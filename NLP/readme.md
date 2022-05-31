CSI5386 project: Depression Detection in Reddit’s COVID-Related Posts.

The **data_collector.py** is used to fetch posts from Reddit. These data are cleaned and preprocessed by **preprocess_reddit_datasets.py**. The processed data are saved in **'../datasets/training_dataset.p'**.

Then the processed training data are split further into training set, validation set, and testing set in **split_dataset.py**. This is to ensure the consistency of the training, validation and testing sets for different models. The datasets are stored under the names:
- x_test.p
- x_train.p
- x_val.p
- y_test.p
- y_train.p
- y_val.p

At this point, we have manually labeled 100 data from the aforementioned **x_test.p**. The manually labeled data are in **experiments/fine_tuning_bert_results/data_manual_labeled_predicted.csv**.

Run **classifier_baseline.py** to train naive Bayes and decision tree models. The results are saved under the folder **experiments/simple_classifier_results**.

Run **fine_tuning_bert.py** to fine-tune and train a Bert model. The results are saved under **experiments/fine_tuning_bert_results**. 
The posts related to COVID-19 are classified by the fine-tuned Bert model, the predictions are stored in **experiments/fine_tuning_bert_results/covid_19_dataset_predicted.csv**.

With the predictions from Bert available, run **LDA_analysis.py** to perform LDA analysis on the COVID-19 data that were labeled as "depressed" by the Bert model. This analysis will find the topics and the results are saved in **experiments/LDA_analysis_results**.

