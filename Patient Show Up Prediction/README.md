# About this project
Bay clinic is a medical centre in Houston that operates with a unique mission of blending research and education with clinical and hospital care. 
The medical center has a huge head force of 25,000 employees, and as a result of the combined effort of those employees, the medical center has been able to handle approximately 3 million visits so far.
In recent times, the hospital was incurring losses despite having the finest doctors available and not lacking scheduled appointments. 
To investigate the reason for the anomaly, a sample data dump of appointments medicalcentre.csv is hereby presented. 
The collected data provides information on the patient’s age, gender, appointment date, various diseases, etc. 
To cut costs, predict if a patient will show up on the appointment day or not

# dataset
This dataset includes 14 features and 110527 recoreds. 

# Feature Engineering
Determine the frequency of distinct values in each feature set 

Initialize a function to plot all features within the dataset to visualize for outliers 

ML algorithm requires the variables to be coded into its equivalent integer codes. Encode the string
categorical values into an integer code 

Conduct variability comparison between features using a correlation matrix & drop correlated features

# Model Development
Develop a Naïve Bayes, SVM, Decision tree classifiers to predict the outcome of the test using Python 

Tune the model using GridSearchCV

# Model Evaluation & Comparison
Compare the results of the Naïve Bayes classifier and SVM with the Decision model according to the following criteria: Accuracy, Sensitivity and Specificity. Identify the model that performed best and worst according to each criterion. 

Carry out a ROC analysis to compare the performance of the Naïve Bayes, SVM model with the Decision Tree model. Plot the ROC graph of the models.

