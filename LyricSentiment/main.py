import glob
import re
import nltk
import random
import string
import numpy as np
import pandas
import collections
import itertools
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.metrics import precision, recall, ConfusionMatrix
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


random.seed(33)
np.set_printoptions(precision=2)
classes = ['Angry', 'Happy', 'Relaxed', 'Sad']

stop_words = set(stopwords.words('english'))
port = PorterStemmer()
tknzr = TweetTokenizer()


# Dataset file lists, 4 total, one for each emotion

# Train Data Ingestion
happy_filelist = glob.glob('./data/Happy/Train/happy*.txt')
angry_filelist = glob.glob('./data/Angry/Train/angry*.txt')
relaxed_filelist = glob.glob('./data/Relaxed/Train/relaxed*.txt')
sad_filelist = glob.glob('./data/Sad/Train/sad*.txt')

# Test Data Ingestion
happy_filelist2 = glob.glob('./data/Happy/Test/happy*.txt')
angry_filelist2 = glob.glob('./data/Angry/Test/angry*.txt')
relaxed_filelist2 = glob.glob('./data/Relaxed/Test/relaxed*.txt')
sad_filelist2 = glob.glob('./data/Sad/Test/sad*.txt')

# Combine Train and Test Data For the Lexicon-Based Analysis
happy_filelist = happy_filelist + happy_filelist2
angry_filelist = angry_filelist + angry_filelist2
relaxed_filelist = relaxed_filelist + relaxed_filelist2
sad_filelist = sad_filelist + sad_filelist2


def read(filelist, tag):
    lyrics = []
    all_words = []

    for f in filelist:
        try:
            with open(f,'r') as file:
                song = file.read()
                file.close()
                song = re.sub(r"(\\n|\\u....|\t)", "", song)
                song = re.sub(r"(\[\d\d:\d\d\.\d\d\])","",song)
                song = song.lower()
                song = nltk.word_tokenize(song)
                #song = tknzr.tokenize(song)
                song = [w for w in song if not w in string.punctuation]
                song = [w for w in song if not w in stop_words]
                song = [port.stem(w) for w in song]
                song_tag = (song, tag)
                lyrics.append(song_tag)

                for word in song:
                    all_words.append(word)
        except:
            break
    return lyrics, all_words

def find_features(song):
    words = set(song)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



happy_lyrics, happy_words = read(happy_filelist, 'happy')
angry_lyrics, angry_words = read(angry_filelist, 'angry')
relaxed_lyrics, relaxed_words = read(relaxed_filelist, 'relaxed')
sad_lyrics, sad_words = read(sad_filelist, 'sad')

data = happy_lyrics + angry_lyrics + relaxed_lyrics + sad_lyrics

all_words = happy_words + angry_words + relaxed_words + sad_words
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:100]

#random.shuffle(data)
#featuresets = [(find_features(song), tag) for (song, tag) in data]

#train_data = featuresets[:320]
#test_data = featuresets[321:]

random.shuffle(happy_lyrics)
random.shuffle(angry_lyrics)
random.shuffle(relaxed_lyrics)
random.shuffle(sad_lyrics)

folds = 5
subset_size = 20
sum = 0
test_truth = []
test_predict = []

happy_prec = 0
angry_prec = 0
relaxed_prec = 0
sad_prec = 0
happy_rec = 0
angry_rec = 0
relaxed_rec = 0
sad_rec = 0

test_truth = []
test_predict = []

for i in range (folds):
    start = int(i*subset_size)
    end = int(start + subset_size)

    #random.shuffle(happy_lyrics)
    test_setHappy = happy_lyrics[start:end]
    train_setHappy = happy_lyrics[:start] + happy_lyrics[end:]

    #random.shuffle(angry_lyrics)
    test_setAngry = angry_lyrics[start:end]
    train_setAngry = angry_lyrics[:start] + angry_lyrics[end:]

    #random.shuffle(relaxed_lyrics)
    test_setRelaxed = relaxed_lyrics[start:end]
    train_setRelaxed = relaxed_lyrics[:start] + relaxed_lyrics[end:]

    #random.shuffle(sad_lyrics)
    test_setSad = sad_lyrics[start:end]
    train_setSad = sad_lyrics[:start] + sad_lyrics[end:]

    training_docs = train_setHappy + train_setAngry + train_setRelaxed + train_setSad
    testing_docs = test_setHappy + test_setAngry + test_setRelaxed + test_setSad

    train_set = [(find_features(song), tag) for (song, tag) in training_docs]

    test_set = [(find_features(song), tag) for (song, tag) in testing_docs]

    model = nltk.NaiveBayesClassifier.train(train_set)
    
    #mnb = SklearnClassifier(MultinomialNB())
    #model = mnb.train(train_set)
    #bnb = SklearnClassifier(BernoulliNB())
    #model = bnb.train(train_set)
    
    #lg = SklearnClassifier(LogisticRegression())
    #model = lg.train(train_set)
    
    #svc = SklearnClassifier(LinearSVC())
    #model = svc.train(train_set)
    
    #nsvc = SklearnClassifier(NuSVC())
    #model = nsvc.train(train_set)
    
    #rf = SklearnClassifier(RandomForestClassifier())
    #model = rf.train(train_set)

    acc = nltk.classify.accuracy(model, test_set)

    sum += acc

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for j, (feats, label) in enumerate(test_set):
        refsets[label].add(j)
        observed = model.classify(feats)
        testsets[observed].add(j)

    happy_prec += precision(refsets['happy'], testsets['happy'])
    angry_prec += precision(refsets['angry'], testsets['angry'])
    relaxed_prec += precision(refsets['relaxed'], testsets['relaxed'])
    sad_prec += precision(refsets['sad'], testsets['sad'])
    happy_rec += recall(refsets['happy'], testsets['happy'])
    angry_rec += recall(refsets['angry'], testsets['angry'])
    relaxed_rec += recall(refsets['relaxed'], testsets['relaxed'])
    sad_rec += recall(refsets['sad'], testsets['sad'])

    test_truth += [s  for (t,s) in test_set]
    test_predict += [model.classify(t) for (t,s) in test_set]
    conf = confusion_matrix(test_truth, test_predict)
    
print(plot_confusion_matrix(conf, classes, normalize=True, title='Normalized Confusion Matrix'))





#Print Averge Metrics
print("Average Accuracy:", (sum/folds)*100)
print("Average Happy Precision:", (happy_prec/folds)*100)
print("Average Angry Precision:", (angry_prec/folds)*100)
print("Average Relaxed Precision:", (relaxed_prec/folds)*100)
print("Average Sad Precision:", (sad_prec/folds)*100)
print("Average Happy Recall:", (happy_rec/folds)*100)
print("Average Angry Recall:", (angry_rec/folds)*100)
print("Average Relaxed Recall:", (relaxed_rec/folds)*100)
print("Average Sad Recall:", (sad_rec/folds)*100)







#                              #
#                              #
#         Lexicon-Based        #
#      Sentiment Analysis      #
#                              #
#                              #

# LSA BOW WPS

# Classifies the emotion of song lyrics based on the similarity of lyrics
# and that class's lexicon.  The lexicon is created with the seed word of 
# the class and is increased in size with synonyms.  Each word in a given
# song lyric is compared to each emotion synonym in the class and the sum
# is taken.  This sum of emotion similarity is calculated for all 4 emotions
# for each test data song.  The maximum of the 4 is chosen as the class
# label for that song. Uses wordnet to build the lexicons.



# Find synonyms of the four class emotions (in bag of words format)

# Relaxed
synonymsR = []
for syn in wn.synsets("relaxed"):
    for l in syn.lemmas():
        synonymsR.append(l.name())

# Relaxed: adding words to increase the lexicon size
for syn in wn.synsets("calm"):
    for l in syn.lemmas():
        synonymsR.append(l.name())

        
# Happy
synonymsH = []
for syn in wn.synsets("happy"):
    for l in syn.lemmas():
        synonymsH.append(l.name())

# Happy: adding "pleasant" to increase the lexicon size
for syn in wn.synsets("pleasant"):
    for l in syn.lemmas():
        synonymsH.append(l.name())

# Happy: adding "laugh" to increase the lexicon size
for syn in wn.synsets("laugh"):
    for l in syn.lemmas():
        synonymsH.append(l.name())
        
# Sad
synonymsS = []
for syn in wn.synsets("sad"):
    for l in syn.lemmas():
        synonymsS.append(l.name())

# adding words to Sad Lexicon to increase size
for syn in wn.synsets("lonely"):
    for l in syn.lemmas():
        synonymsS.append(l.name())


# Angry
synonymsA = []
for syn in wn.synsets("angry"):
    for l in syn.lemmas():
        synonymsA.append(l.name())

# Angry: adding words to increase the lexicon
for syn in wn.synsets("hate"):
    for l in syn.lemmas():
        synonymsA.append(l.name())

for syn in wn.synsets("kill"):
    for l in syn.lemmas():
        synonymsA.append(l.name())

#synonymsA[0] # gives the 1st element in the list

# Get unique values only
synonymsA = set(synonymsA) 
synonymsR = set(synonymsR)
synonymsH = set(synonymsH)
synonymsS = set(synonymsS)

# Convert sets back to lists
synonymsA = list(synonymsA) 
synonymsR = list(synonymsR)
synonymsH = list(synonymsH)
synonymsS = list(synonymsS)


#synonymsH[0]
print(synonymsH)
print(synonymsR)
print(synonymsS)
print(synonymsA)

# Specify the data that we will use to test the LSA
testdata = data  #testdata[1][1] # Returns the pre-determined label for the song

# Create a dataframe to store the test data's emotions similarity sum 
# scores for each of the four emotions. Rows = songs and 4 cols = emotion, one is for max class label 
# similarity sums. df#rows = #rows in the test data.

df1 = pandas.DataFrame(index=np.arange(len(testdata)), columns = ['happy', 'relaxed', 'sad', 'angry', 'MaxClass', 'ActualClass'])

# this is the class label: testdata[songIndex][1]
# need to add this to the matrix for each index of testdata[index][1] at df1[index, 5]
  
# add "actual" class labels to the matrix for each song
for songIndex in range(len(testdata)):
    df1.iloc[songIndex, 5] = testdata[songIndex][1]

#df1
    
# Compare each lyric word in a song to every word in the Synonym list 
#  for each emotion

# HAPPY COLUMN CALCULATIONS

#this loop will assess every song
songind = 0  #j will be used as the row value for inputting sim sums into the df

x = len(testdata)  #number of items in the test set

#loop through every song:
for i in range(x):
    #songind+=1
    firstsong = testdata[songind][0]

    sum = 0  # initialize the sum value, per song, per emotion. If the simw1w2 is not 0, i.e. not "None", we want to keep track of it

    for w in firstsong:
        #if w1 is not null, then we do the rest of this cell....
        if wn.synsets(w):
            w1 = wn.synsets(w)[0]
            #print('w1')
            #print(w1)
            # For each of the synonyms
            for j in range(len(synonymsH)):
                w2 = synonymsH[j]
                w2 = wn.synsets(w2)[0]
                #print('w2')
                #print(w2)
                w1w2sim = w1.wup_similarity(w2)
                #print(w1w2sim)
                if w1w2sim != None:
                    #if w1w2sim > 0.7:
                    sum += w1w2sim
                    sum = sum / len(synonymsH) #normalize the sum by dividing by the number of synonymns
                    sum = sum / len(firstsong) #normalize the sum by dividing by the number of words in the song
                        #print('sum')
                        #print(sum)   # this is the sum for the first song's happy similarities
    df1.iloc[songind,0] = sum
    songind += 1
    
# RELAXED CALCULATIONS

#this loop will assess every song
songind = 0  #j will be used as the row value for inputting sim sums into the df

for i in range(x):
    #songind+=1
    firstsong = testdata[songind][0]

    sum = 0  # initialize the sum value, per song, per emotion. If the simw1w2 is not 0, i.e. not "None", we want to keep track of it

    for w in firstsong:
        #if w1 is not null, then we do the rest of this cell....
        if wn.synsets(w):
            w1 = wn.synsets(w)[0]
            #print('w1')
            #print(w1)
            # For each of the synonyms of Relaxed
            for j in range(len(synonymsR)):
                w2 = synonymsR[j]
                w2 = wn.synsets(w2)[0]
                #print('w2')
                #print(w2)
                w1w2sim = w1.wup_similarity(w2)
                #print(w1w2sim)
                if w1w2sim != None:
                    #if w1w2sim > 0.7:
                    sum += w1w2sim
                    sum = sum / len(synonymsR) #normalize the sum by dividing by the number of synonymns
                    sum = sum / len(firstsong) #normalize the sum by dividing by the number of words in the song
                        #print('sum')
                        #print(sum)   # this is the sum for the first song's Relaxed similarities
    df1.iloc[songind,1] = sum
    songind += 1
    
# SAD CALCULATIONS

#this loop will assess every song
songind = 0  #j will be used as the row value for inputting sim sums into the df

for i in range(x):
    #songind+=1
    firstsong = testdata[songind][0]

    sum = 0  # initialize the sum value, per song, per emotion. If the simw1w2 is not 0, i.e. not "None", we want to keep track of it

    for w in firstsong:
        #if w1 is not null, then we do the rest of this cell....
        if wn.synsets(w):
            w1 = wn.synsets(w)[0]
            #print('w1')
            #print(w1)
            # For each of the synonyms of Sad
            for j in range(len(synonymsS)):
                w2 = synonymsS[j]
                w2 = wn.synsets(w2)[0]
                #print('w2')
                #print(w2)
                w1w2sim = w1.wup_similarity(w2)
                #print(w1w2sim)
                if w1w2sim != None:
                    #if w1w2sim > 0.7:
                    sum += w1w2sim
                    sum = sum / len(synonymsS) #normalize the sum by dividing by the number of synonymns
                    sum = sum / len(firstsong) #normalize the sum by dividing by the number of words in the song
                        #print('sum')
                        #print(sum)   # this is the sum for the first song's Sad similarities
    df1.iloc[songind,2] = sum
    songind += 1
    
# ANGRY CALCULATIONS

#this loop will assess every song
songind = 0  #j will be used as the row value for inputting sim sums into the df

x = len(testdata)
for i in range(x):
    #songind+=1
    firstsong = testdata[songind][0]

    sum = 0  # initialize the sum value, per song, per emotion. If the simw1w2 is not 0, i.e. not "None", we want to keep track of it

    for w in firstsong:
        #if w1 is not null, then we do the rest of this cell....
        if wn.synsets(w):
            w1 = wn.synsets(w)[0]
            #print('w1')
            #print(w1)
            # For each of the synonyms of Angry
            for j in range(len(synonymsA)):
                w2 = synonymsA[j]
                w2 = wn.synsets(w2)[0]
                #print('w2')
                #print(w2)
                w1w2sim = w1.wup_similarity(w2)
                #print(w1w2sim)
                if w1w2sim != None:
                    #if w1w2sim > 0.7:
                    sum += w1w2sim
                    sum = sum / len(synonymsA) #normalize the sum by dividing by the number of synonymns
                    sum = sum / len(firstsong) #normalize the sum by dividing by the number of words in the song
                        #print('sum')
                        #print(sum)   # this is the sum for the first song's Angry similarities
    df1.iloc[songind,3] = sum
    songind += 1
    
# Find the max similarity value of each row and assign the class of that
# value as the Maximum Class Output class label

indx = 0

for i in range(x):
    #print(df1.iloc[indx].argmax())
    df1.iloc[indx, 4] = df1.iloc[indx, 0:4].argmax()
    #df1.iloc[indx, 4] = df1.iloc[indx].argmax()
    
    indx += 1

df1.iloc[0:10,:]  #Showing a portion of df1: the final collection of calculations and predictions

# Calculate Precision and Recall for each class

preds = list(df1.iloc[:, 4])    #predicted class labels to feed into the confusion matrix
actuals = list(df1.iloc[:, 5])  #actual class labels

lsa_cm = confusion_matrix(actuals, preds)
#cm.print_stats()

# Plot the resulting confusion matrix

print(plot_confusion_matrix(lsa_cm, classes, normalize=True, title='Normalized Confusion Matrix'))
#skplt.metrics.plot_confusion_matrix(actuals, preds, normalize=True)
#plt.show()


