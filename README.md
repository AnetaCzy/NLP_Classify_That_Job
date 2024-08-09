# NLP_Classify_That_Job
NLP project, developed as part of a Bachelor, thesis that explores the implementation of three machine learning algorithms—Naïve Bayes, Support Vector Machine (SVM) with Stochastic Gradient Descent (SGD), and Logistic Regression—to classify job posting descriptions based on textual data. The research involves preprocessing job descriptions, extracting relevant features, and applying these algorithms to categorize job postings into specific occupation classes. The project aims to evaluate and compare the effectiveness of these algorithms in the context of job classification, providing insights into their performance for text-based data analysis.

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

