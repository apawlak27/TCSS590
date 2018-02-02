#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:31:03 2018

@author: alexandrapawlak
"""

import glob
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

# Dataset file lists
neg_filelist = glob.glob('./neg/*.txt')
pos_filelist = glob.glob('./pos/*.txt')

# Read Read Volg text files, change all words to lowercase, and remove punctuation
# Returns word counts across all docs 
def readfiles(filelist):
    wordCount = Counter()
    allText = []
    
    for f in filelist:
        file = open(f, "r")
        text = file.read().lower()
        file.close()
        text = re.sub("<br />", " ", text)
        text = re.sub('[^a-z\']+', " ", text)
        text = text.split()
        #text = [word for word in text if word not in stopwords.words('english')]
        allText.append(text)
        wordCount.update(text)
    
    return allText, wordCount

# Returns total count of words in given dictionary
def count(words):
    return sum(words.values())

# Returns number of unique words in given dictionary
def unique(pos, neg):
    words = Counter()
    words = {**pos, **neg}
    return words

# Returns bigrams for given list of text
def bigram(words):
    bigrams = []
    for n in words:
        split = nltk.bigrams(n)
        bigrams.extend(split)
    return bigrams

# Returns trigrams for given list of text
def trigram(words):
    trigrams = []
    for n in words:
        split = nltk.trigrams(n)
        trigrams.extend(split)
    return trigrams

# Returns Frequency of given list of bigrams or trigrams
def freq(words):
    return nltk.FreqDist(words)

# Returns 10 most frequent bigrams or trigrams
def topTen(freq):
    for k,v in freq.most_common(10):
        print(k,v)
        
# Lists of positive and negative words and their counts     
posWords, posCounter = readfiles(pos_filelist)
negWords, negCounter = readfiles(neg_filelist)

# Total number of words in database
totalWords = count(posCounter) + count(negCounter)

# Total number of unique words in database
uniqueWords = len(unique(posCounter,negCounter))

print("Total number of word tokens in the database: %d" % totalWords)
print("Total number of uniqe words in the dataset: %d" % uniqueWords)
print("")

# Birgrams and Trigrams for positive and negative word banks
negBigrams = bigram(negWords)
posBigrams = bigram(posWords)

negTrigrams = trigram(negWords)
posTrigrams = trigram(posWords)

# Frequency of Bigrams and Trigrams in postive and negative word banks
# Prints top ten Bigrams and Trigrams for each
posBiFreq = freq(posBigrams)
print("Top 10 Positive Bigrams:")
topTen(posBiFreq)

print("")

negBiFreq = freq(negBigrams)
print("Top 10 Negative Bigrams:")
topTen(negBiFreq)

print("")

posTriFreq = freq(posTrigrams)
print("Top 10 Positive Trigrams:")
topTen(posTriFreq)

print("")

negTriFreq = freq(negTrigrams)
print("Top 10 Negative Trigrams:")
topTen(negTriFreq)

print("")

# All text from both positive and negative files
allText = posWords + negWords

# Trigrams from all text
allTrigrams = trigram(allText)

# Frequency of Trigrams from all text
allTriFreq = freq(allTrigrams)

# Function takes a series of 3 words and computes the probability 
# of the third word using the Trigram Language Model with smoothing
# p(w3|w1, w2) = (count(w1, w2, w3) + 1) / ((total number of trigrams) + |V|)
# Smoothing allows for probabilities of word series that have not been seen before
# |V| is the vocabulary size.
# Returns the probability of the third word given the first two

def probability(w1, w2, w3):
    count = allTriFreq[w1, w2, w3]
    prob = (count + 1)/(len(allTrigrams)+uniqueWords)
    return prob

# Five test cases of the probability function
print(probability('the', 'amazing', 'performance'))
print(probability('a', 'couple', 'of'))
print(probability('showcase', 'his', 'wife'))
print(probability('in', 'the', 'theater'))
print(probability('had', 'struck', 'it'))
