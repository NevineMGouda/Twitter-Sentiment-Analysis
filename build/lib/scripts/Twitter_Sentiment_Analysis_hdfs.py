#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# IMPORTS
####################################################
import sys
import emot
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pytypo
except:
    logger.error("An Error importing library: pytypo")
    logger.error("Please try running :'sudo pip install pytypo' OR 'sudo pip3 install pytypo+' and try again")
    sys.exit(0)

try:
    from pyspark import SparkContext, SparkConf
except:
    logger.error("An Error importing library: pyspark\n")
    logger.error("If Linux/Unix Enviroment Please try:\n" \
          "\t 1- Installing Apache Spark using this link: https://spark.apache.org/downloads.html\n " \
          "\t 2- Running: 'sudo pip install pyspark' OR 'sudo pip3 install pyspark+'\n" \
          "And try again.\n")

    logger.error("If MacOS Enviroment Please try:\n" \
          "\t 1- Installing homebrew by following: https://brew.sh/\n" \
          "\t 2- Running: 'brew install apache-spark' \n" \
          "\t 3- Running: 'sudo pip install pyspark' OR 'sudo pip3 install pyspark+' \n" \
          "And try again.")
    sys.exit(0)
try:
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
except:
    logger.error("An Error importing library: nltk")
    logger.error("Please try running :'sudo pip install -U nltk' And 'sudo pip install -U numpy' and try again")
    sys.exit(0)

import nltk
import time
import string
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import SVMWithSGD, LogisticRegressionWithLBFGS
#sudo -u hdfs hadoop fs -mkdir /hdf_data
#sudo -u hdfs hadoop fs -put myfile.txt /hdf_data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# END OF IMPORTS
####################################################
# TODO: Change from random Split to normal split and CV
# TODO: try DT and SVM algorithms
# TODO: run ready libraries and compare results

# START OF GLOBAL VARIABLES
####################################################
PUNCTUATION = [i for i in string.punctuation]
STOPWORDS = set(stopwords.words('english'))
HEADER = []
PS = PorterStemmer()

conf = SparkConf().setAppName("myFirstApp").setMaster("local")
SC = SparkContext(conf=conf)
HTF = HashingTF(50000)
# END OF GLOBAL VARIABLES
####################################################

# Fix tweet if split by comma
def line_fixer(line, col_count):
    if len(line) > col_count:
        return line[:col_count-1] + [",".join(line[col_count-1:])]
    return line


# Ignore unwanted columns like the tweet ID (column 0) and SentimentSource (column 2)
def remove_unwanted_col(line, sentinement_index, sentinement_text_index):
    return line[sentinement_index], line[sentinement_text_index]


def lower_case(tweet):
    return [word.lower() for word in tweet]


def remove_punc_keep_emoj(tweet):
    emoticons = emot.emoticons(tweet)
    punct_removed = []
    for word in tweet:
        word_list = []
        for char in word:
            if char not in string.punctuation:
                word_list.append(char)

        word_list = ''.join(word_list)
        if len(word_list) == 0:
            continue
        punct_removed.append(word_list)
    if len(emoticons) != 0:
        for record in emoticons:
            emo = record['value']
            punct_removed.append(emo)

    return punct_removed


def stem_words(word):
    return PS.stem(word=word)


def remove_stop_words(word):
    if word not in STOPWORDS:
        return word


def pre_process(rdd):
    # Tokenize each line
    # rdd = rdd.map(lambda (sentiment, tweet): (sentiment, word_tokenize(tweet)))
    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, tweet.strip().split(" ")))

    # Lowercase each word in tweet and return and rdd equivlant to (0,['this', 'is', 'a', 'lowercased','tweet'])
    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, lower_case(tweet=tweet)))

    # Remove punctuation from a tweet.
    # Example :(0,["is,","so","sad","for","my","APL","friend","............."]) should be mapped to
    #          (0,["is","so","sad","for","my","APL","friend"])
    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, remove_punc_keep_emoj(tweet=tweet)))
    # Stem words to their original. For example: "missing" or "missed" -> "miss"
    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, [stem_words(word=word) for word in tweet]))
    # Remove stop words such as: a, I, and, all, once, etc.
    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, [word for word in tweet if word not in STOPWORDS]))

    # Map elongated words with a shorter version, with only 3 letters of the repeated words
    # Example: cooooollllll is mapped to cooolll
    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, [pytypo.cut_repeat(word, 3) for word in tweet]))
    return rdd


def get_word_ratio(rdd1, word):
    stemmed_word = PS.stem(word=word)
    rdd1 = rdd1.filter(lambda (sentiment, tweet): stemmed_word in tweet)
    rdd1 = rdd1.map(lambda (sentiment, tweet): (sentiment,1))
    tdd1 = rdd1.groupByKey().map(lambda (sentiment, count) : (sentiment, sum(count)))
    for record in tdd1.collect():
        logger.info("__TSA__: The word" + word + "occured" + record[1] + "times in Sentiments of label" + record[0])


def SA_training(input_filename):

    # Import full dataset of newsgroup posts as text file
    # input_filename = "hdfs://localhost:9000/user/nevinemgouda/small.csv"
    try:
        rdd = SC.textFile(input_filename)
        logger.info("__TSA__: Successfully loaded the data from: " + input_filename)
    except:
        logger.error("__TSA__: Something went wrong while importing from: " + input_filename)
        sys.exit(0)

    rdd = rdd.map(lambda line: line.split(","))
    HEADER = rdd.take(1)[0]
    sentinement_index = HEADER.index("Sentiment")
    sentinement_text_index = HEADER.index("SentimentText")
    # Remove the header from the rdd
    rdd = rdd.filter(lambda line: line != HEADER and len(line) >= 4)

    # Fix tweet if it contained "," and removed while splitting

    rdd = rdd.map(lambda line: line_fixer(line, len(HEADER)))
    # Return only the label and the tweet and ignore other columns.
    # now rddd example [[1,"This is the first positive tweet"], [0, "This is the first negative tweet"]]
    rdd = rdd.map(lambda line: remove_unwanted_col(line, sentinement_index, sentinement_text_index))
    rdd = pre_process(rdd)

    # Hash the data and split it into taining and testing data for the algorithms models
    data_hashed = rdd.map(lambda (sentiment, tweet): LabeledPoint(sentiment, HTF.transform(tweet)))
    train_hashed, test_hashed = data_hashed.randomSplit([0.7, 0.3])

    # Train the SVM with SGD model using the hashed trained data
    logger.info("__TSA__: Training The Logistic Regression Model...")
    lr_model = LogisticRegressionWithLBFGS.train(train_hashed, 100)
    # dt_model = DecisionTree.trainClassifier(data=train_hashed, numClasses=2, categoricalFeaturesInfo={}, maxDepth=5, maxBins=64)
    prediction_and_labels = test_hashed.map(lambda point: (lr_model.predict(point.features), point.label))
    correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
    accuracy = correct.count() / float(test_hashed.count())
    logger.info("__TSA__: Logistic Regression correctly classified the tweets with an accuracy of " + str(
        accuracy * 100) + "%.")

    # Train the SVM with SGD model using the hashed trained data
    logger.info("__TSA__: Training The SVM with SGD Model...")
    svm_model = SVMWithSGD.train(data=train_hashed, iterations=100)
    prediction_and_labels = test_hashed.map(lambda point:  (svm_model.predict(point.features), point.label))
    correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
    accuracy = correct.count() / float(test_hashed.count())
    logger.info("__TSA__: SVM with Stochastic Gradient Descent correctly classified the tweets with an accuracy of " + str(
        accuracy * 100) + "%.")

    # Train the Naive Bayes model using the hashed trained data
    logger.info("__TSA__: Training The Naive Bayes Model...")
    nb_model = NaiveBayes.train(train_hashed, lambda_=1.0)
    prediction_and_labels = test_hashed.map(lambda point: (nb_model.predict(point.features), point.label))
    correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
    accuracy = correct.count() / float(test_hashed.count())
    logger.info("Naive Bayes correctly classified the tweets with an accuracy of " + str(accuracy * 100) + "%.")


    return nb_model, svm_model, lr_model, rdd


def main():
    input_filename = raw_input("Please enter the filename (with extension)"
                               " that you want to test: ")
    if input_filename.lower() == "d":
        input_filename = "hdfs://localhost:9000/user/nevinemgouda/small.csv"
    logger.info("__TSA__: We are currently training the models with the input data... Please wait :)")
    nb_model, svm_model, lr_model, rdd = SA_training(input_filename)
    time.sleep(0.5)
    while True:
        input_tweet = raw_input("\nEnter a tweet to classify or just Q to quit:\n")
        if input_tweet.lower() == "q":
            logger.info("__TSA__: Bye Bye :)")
            break
        if len(input_tweet.strip()) == 0:
            continue
        tweet = [(1, input_tweet)]
        input_rdd = SC.parallelize(tweet)
        input_rdd = pre_process(input_rdd)
        input_hashed = input_rdd.map(lambda (sentiment, tweet): LabeledPoint(sentiment, HTF.transform(tweet)))
        nb_prediction = input_hashed.map(lambda point: nb_model.predict(point.features))
        svm_prediction = input_hashed.map(lambda point: svm_model.predict(point.features))
        lr_prediction = input_hashed.map(lambda point: lr_model.predict(point.features))
        if lr_prediction.collect()[0] == 0:
            logger.info("__TSA__: We predict that you entered a NEGATIVE tweet using Logistic Regression!")
        else:
            logger.info("__TSA__: We predict that you entered a POSITIVE tweet using Logistic Regression!")

        if svm_prediction.collect()[0] == 0:
            logger.info("__TSA__: We predict that you entered a NEGATIVE tweet using SVM with Stochastic Gradient Descent!")
        else:
            logger.info("__TSA__: We predict that you entered a POSITIVE tweet using SVM with Stochastic Gradient Descent!")

        if nb_prediction.collect()[0] == 0:
            logger.info("__TSA__: We predict that you entered a NEGATIVE tweet using Naive Bayes!")
        else:
            logger.info("__TSA__: We predict that you entered a POSITIVE tweet using Naive Bayes!")

        time.sleep(1)
        word = raw_input("__TSA__: Please enter a word to see how much it is present in the input data or 'S' to skip: ")
        if word.lower() == 's':
            continue
        get_word_ratio(rdd, word=word)

if __name__ == "__main__":
    main()