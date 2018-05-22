#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import nltk
import string
import sys


import re
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pyspark import SparkContext
import pytypo
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

# TODO: Automatically install missing libraries
# TODO: Change from random Split to normal split and CV to compare the following:
# TODO: emoticons fix
# TODO: try  KNN or SVM algorithms
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
EMOTICONS = [u":)", u":(",u":')",]
HEADER = []
PS = PorterStemmer()
SC = SparkContext()
HTF = HashingTF(50000)

# Fix tweet if split by comma
def line_fixer(line, col_count):
    if len(line) > col_count:
        return line[:col_count-1] + [",".join(line[col_count-1:])]
    return line

# Ignore unwanted columns like the tweet ID (column 0) and SentimentSource (column 2)
def remove_unwanted_col(line):
    return line[1], line[3]

def lower_case(tweet):
    return [word.lower() for word in tweet]

def remove_punctuation(tweet):
    punct_removed = []
    for word in tweet:
        word_list = []
        for char in word:
            if char not in string.punctuation:
                word_list.append(char)

        word_list = ''.join(word_list)
        if word_list != word:
            for emoticon in EMOTICONS:
                if emoticon in word:
                    punct_removed.append(emoticon)
        if len(word_list) == 0:
            continue
        punct_removed.append(word_list)
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

    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, remove_punctuation(tweet=tweet)))
    # Stem words to their original. For example: "missing" or "missed" -> "miss"
    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, [stem_words(word=word) for word in tweet]))
    # Remove stop words such as: a, I, and, all, once, etc.
    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, [word for word in tweet if word not in STOPWORDS]))

    # Map elongated words with a shorter version, with only 3 letters of the repeated words
    # Example: cooooollllll is mapped to cooolll
    rdd = rdd.map(lambda (sentiment, tweet): (sentiment, [pytypo.cut_repeat(word, 3) for word in tweet]))
    return rdd

def get_word_ratio(rdd1, word):
    word = PS.stem(word=word)
    rdd1 = rdd1.filter(lambda (sentiment, tweet): word in tweet)
    rdd1 = rdd1.map(lambda (sentiment, tweet): (sentiment,1))
    #[1 for tweet_word in tweet if tweet_word==word]
    rdd1 = rdd1.groupByKey().map(lambda (sentiment, count):(sentiment, sum(count)))

def SA_training():
    # Initialize a SparkContext
    input_filename = "small.csv"

    # Import full dataset of newsgroup posts as text file
    rdd = SC.textFile(input_filename)
    rdd = rdd.map(lambda line: line.split(","))
    HEADER = rdd.take(1)[0]
    # Remove the header from the rdd
    rdd = rdd.filter(lambda line: line != HEADER and len(line) >= 4)

    # Fix tweet if it contained "," and removed while splitting

    rdd = rdd.map(lambda line: line_fixer(line, len(HEADER)))
    # Return only the label and the tweet and ignore other columns.
    # now rddd example [[1,"This is the first positive tweet"], [0, "This is the first negative tweet"]]
    rdd = rdd.map(remove_unwanted_col)
    rdd = pre_process(rdd)

    get_word_ratio(rdd, word="hapy")
    data_hashed = rdd.map(lambda (sentiment, tweet): LabeledPoint(sentiment, HTF.transform(tweet)))
    train_hashed, test_hashed = data_hashed.randomSplit([0.7, 0.3])
    model = NaiveBayes.train(train_hashed, lambda_=7.0)
    prediction_and_labels = test_hashed.map(lambda point: (model.predict(point.features), point.label))
    correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
    accuracy = correct.count() / float(test_hashed.count())
    print "Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time"

    return model

def main():
    print "We are currently training the model with our data... Please wait :)"
    model = SA_training()
    while True:
        input_tweet = raw_input("Enter a tweet to classify or just Q to quit:")
        if input_tweet.lower() == "q":
            print "Bye Bye :)"
            break
        tweet = [(1, input_tweet)]
        input_rdd = SC.parallelize(tweet)
        input_rdd = pre_process(input_rdd)
        input_hashed = input_rdd.map(lambda (sentiment, tweet): LabeledPoint(sentiment, HTF.transform(tweet)))
        prediction_and_labels = input_hashed.map(lambda point: (model.predict(point.features), point.label))
        if prediction_and_labels.take(2)[0][0] == 0:
            print "We predict that you entered a NEGATIVE tweet"
        else:
            print "We predict that you entered a POSITIVE tweet"

main()