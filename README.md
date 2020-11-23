# hateSpeechDetectionTwitter
Authors: Shruthi P

Summary: This is a project on automated detection of hate speech on Twitter using data collected by Zeerak Waseem. We have reproduced the baseline (Pinkesh Badjatiya) results.
We have also inlcuded the other deep learning models like BiLSTM and hybrid models for deep learning feature extarction and applied the logistic regression for classification.
We have extended the experiments by combining the deep learning features with other semantic feature like TF-IDF to form hybrid features.

Dataset
We have used the twitter dataset which contains the actual tweets for the IDs given by Zeerak and Waseem , it is in the file HateDataRSN_16k. Tweets are labelled as either Racism, Sexism or None
(Original Dataset can also be downloaded from https://github.com/zeerakw/hatespeech. Contains tweet id's and corresponding annotations. Use your favourite tweet crawler and download the data and place the tweets in the csv file.)  

Requirements

Keras,
Tensorflow / Theano,
Gensim,
xgboost,
NLTK,
Sklearn,
Numpy

Instructions to run:
 You can check the file replicationCommands.doc for running the baseline methods.
 You can also check the Jupyter Notebooks for running our methods. 
