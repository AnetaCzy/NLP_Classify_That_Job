# NLP_Classify_That_Job
NLP project, developed as part of a Bachelor, thesis that explores the implementation of three machine learning algorithms—Naïve Bayes, Support Vector Machine (SVM) with Stochastic Gradient Descent (SGD), and Logistic Regression—to classify job posting descriptions based on textual data. The research involves preprocessing job descriptions, extracting relevant features, and applying these algorithms to categorize job postings into specific occupation classes defined by The International Standard Classification of Occupations (ISCO - a statistical framework that organizes jobs into a clearly defined set of groups according to the tasks and duties undertaken in the job). The project aims to evaluate and compare the effectiveness of these algorithms in the context of job classification, providing insights into their performance for text-based data analysis.

## Project Goal
The main objective of this study was to develop a classifier that automatically assigns single-digit occupation codes to job advertisement content.

## Research Objectives
  C1: Identify which algorithm among those studied provides the most accurate classification.
  
  C2: Prepare a training and test set containing input data in the form of occupation descriptions and their associated 6-digit codes.
  
  C3: Train a model based on new data to verify the assumptions derived from the test models.

## Data Sources
Central Job Offers Database (CBOP - Centralna Baza Ofert Pracy)

Manually coded data collected by the Institute for Educational Research in Warsaw (IBE - Instytut Badań Edukacyjnych w Warszawie)

## Methods
Language: Python (numpy, pandas, regex, nltk.corpus, matplotlib.pyplot, seaborn, sklearn)

Techniques: Pipelines, CountVectorizer, Tf-idf Transformer

Algorithms:
  1) Naive Bayes Classifier
  2) Support Vector Machine (SVM) with Stochastic Gradient Descent (SGD)
  3) Logistic Regression

## Built Classifiers
For CBOP data:
  Naive Bayes classifier for multinomial models
  Linear SVM classifier
  Logistic classification based on a logistic regression model
For CBOP data tested on IBE data:
  Naive Bayes classifier for multinomial models
  Linear SVM classifier
  Logistic classification based on a logistic regression model

## Data Distribution
![image](https://github.com/user-attachments/assets/2c55e31f-a1d2-4c28-a208-e7596f91e264)

## Classification Results
Classifiers were built and evaluated on two datasets (CBOP and IBE), yielding the following results:

For CBOP data:
  The best result was achieved using logistic regression with a weighted average f-1 score of 86%.
  The best predictions were for classes 5, 9, and 4, while the worst were for class 3.
  For IBE data:

The best result was also obtained using logistic regression with a weighted average f-1 score of 74%.
The best-predicted classes were 2, 5, and 9, with the worst being class 3.

![image](https://github.com/user-attachments/assets/e6ef8654-c239-40be-9ace-1d38da47734a)

## Thesis summary
The aim of this thesis was to find the best model for the classification of occupations and to match job descriptions with the right job categories based on KZiS standards kept by the Central Statistical Office (pl. GUS). It will contribute to the continuation of work towards the automation of the classification of occupations for the purposes of statistics, and the data obtained from its results can be analyzed in terms of adjusting the labor market to the economic development of the country.

The study was conducted on data from the Central Job Offers Database (CBOP) and different web pages with job postings collected by the Institute for Educational Research. Before creating and training our models, the input data was thoroughly cleaned to minimize possible interference and optimize the accuracy of the result.
We analyzed three algorithms that work well with natural language processing and are commonly used for text classification: Naive Bayes, Support-Vector Machine with Stochastic Gradient Descent (SGD) and Logistic Regression. 

After successfully running our models on the first dataset, we decided to run a test on new data. We got a sample of 10,062 manually labeled job offers and loaded them to our pre-trained model. It got us roughly 1,43% to predict out of 703887 trained samples. We performed the same data-cleaning process as before and compared the results.

The best fit was achieved by using a logistic regression model with an overall weighted average f1-score of 74%. Following up with 68% of SVM weighted average f1-score and the least fitted model of Naive Bayes (58%). We can also see that in all three models, the best-predicted classes are no. 2 (Specialists), no. 7 (Industrial
workers and Craftsmen) and no. 5 (Service and Sales workers). These were the classes with the highest number of samples in the dataset and the highest percentage representation of the ratio of the test class to the trained class. The least predicted classes were class no. 1 (that weren't even included in Naive Bayes model) and class no.3, both commonly mistakenly fitted in the first category.

The conclusions that can be drawn from our observations are that, although working with unbalanced data can give relatively good outcomes, we have to be careful while interpreting it, because of the possible bias. Also, a bigger training sample of one of the classes usually positively influences the result for this class but also decreases the accuracy of prediction for other classes. 

## Possible Future Improvements
For further investigation regarding the improvement of the model's accuracy, there are a few topics yet to be considered. First of all, text stemming and lemmatization during the data cleaning process. There are yet not many algorithms developed to handle these tasks in the Polish language, which is very complex from a grammatical point of view and it is also highly inflectional. Stemming is the truncation of all kinds of prefixes and suffixes to get to the unchanging "core" representing the word. The root itself does not necessarily have to be the correct word, which may also lead to anomalies while analyzing text, while lemmatization is reducing a word to its basic form. The most famous algorithm used for lemmatization in polish is Morfeusz, but it does not always correspond correctly to a given problem with the word.

Another topic to think about would be a mechanism to select characteristic features, based on which the classifier could be taught. A good step toward improving our model could be also adding word filtering. That is to select subsets of words that appear in certain classes. The usual filtering mechanism is used to remove redundant information (so-called noise). Such words for the classifier do not carry any useful knowledge and may reduce the accuracy of the classification, as well as increase the time of the classification process itself. The easiest approach to this problem would be adding those words to the created stop-words list and remove them from jobs descriptions. As a high impact on results had the distribution of samples in given classes, there are some methods to work with, that could eliminate this problem. For instance, to overcome an imbalanced dataset, there is a bootstrap method that iteratively resamples a dataset with replacements or different Synthetic Minority Oversampling Techniques, known as SMOTE, to balance our data. 
