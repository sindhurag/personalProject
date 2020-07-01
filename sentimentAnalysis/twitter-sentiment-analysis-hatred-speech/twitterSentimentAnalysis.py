'''
Dataset includes racist/non-racist comments from twitter
using naive-bayes classification, test the accuracy of prediction of test data

first - clean train and test data to include only text
Intention: Compare results with stemmed and unstemmed data

From Kaggle data set:
The objective of this task is to detect hate speech in tweets. For the sake of simplicity,
we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it.
So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist
and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.

Content

Label = 1 -> hate/racist
label = 0 -> non-hate/non-racist

So sentiment = <0 -> label = 1
sentiment = >0 -> label = 0


'''

import pandas as pd

import re
import nltk

# Use this cell to download all the required corpora first. Then, comment out this
# block of code.
#print("Downloading corpora...")
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')
#print("Corpora download complete.")

#nltk.download('vader_lexicon')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



clean_train = "/Users/iamsgqa/Documents/Automation/Experiments_org/Kaggle/sentimentAnalysis/twitter-sentiment-analysis-hatred-speech/data/clean_train.csv"
train_data = "/Users/iamsgqa/Documents/Automation/Experiments_org/Kaggle/sentimentAnalysis/twitter-sentiment-analysis-hatred-speech/data/train.csv"
clean_test = "/Users/iamsgqa/Documents/Automation/Experiments_org/Kaggle/sentimentAnalysis/twitter-sentiment-analysis-hatred-speech/data/clean_test.csv"
test_data = "/Users/iamsgqa/Documents/Automation/Experiments_org/Kaggle/sentimentAnalysis/twitter-sentiment-analysis-hatred-speech/data/test.csv"
sentiment_train = "/Users/iamsgqa/Documents/Automation/Experiments_org/Kaggle/sentimentAnalysis/twitter-sentiment-analysis-hatred-speech/data/sentiment_train.csv"
sentiment_train_stemmed = "/Users/iamsgqa/Documents/Automation/Experiments_org/Kaggle/sentimentAnalysis/twitter-sentiment-analysis-hatred-speech/data/sentiment_train_stemmed.csv"
prediction_test = "/Users/iamsgqa/Documents/Automation/Experiments_org/Kaggle/sentimentAnalysis/twitter-sentiment-analysis-hatred-speech/data/prediction_test.csv"
prediction_test_stemmed = "/Users/iamsgqa/Documents/Automation/Experiments_org/Kaggle/sentimentAnalysis/twitter-sentiment-analysis-hatred-speech/data/prediction_test_stemmed.csv"

def cleanData(df):
    df['clean tweet'] = df['tweet'].replace(r'[^a-zA-Z]+', ' ', regex=True)
    df['clean tweet'] = df['clean tweet'].str.lower()
    #df['clean tweet'] = [re.sub(r"[^a-zA-Z]+", ' ', k) for k in df['tweet'].split("\n")]

    return(df)

def getSA(df):
    neg = []
    pos = []
    neu = []
    comp = []
    senti = []
    sid = SentimentIntensityAnalyzer()
    sentiment = 0
    #df_sample = df.head()
    for index, row in df.iterrows():
        ss = sid.polarity_scores(row['clean tweet'])

        neg.append(ss['neg'])
        pos.append(ss['pos'])
        neu.append(ss['neu'])
        comp.append(ss['compound'])
        if (ss['compound'] >= 0):
            sentiment = 0
        else:
            sentiment = 1
        senti.append(sentiment)

    df['Neg'] = neg
    df['Pos'] = pos
    df['Neu'] = neu
    df['Compound'] = comp
    df['sentiment'] = senti

    return(df)

def getPrediction(text):

    la = []
    sa = []
    la_stemmed = []
    sa_stemmed = []

    t = text.to_list()

    counts = count_vect.transform(t)
    la = clf_label.predict(counts)
    sa = clf_senti.predict(counts)

    la_stemmed = clf_label_stemmed.predict(counts)
    sa_stemmed = clf_label_stemmed.predict(counts)

    df['label_predict'] = la
    df['sentiment_predict'] = sa
    df['label_predict_stemmed'] = la_stemmed
    df['sentiment_predict_stemmed'] = sa_stemmed

    return(df)

# For all the words in the token list, stem the words
def stemWords(tokens):
    stemmer = PorterStemmer()

    stemmedTweet = ''

    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmedTweet = ' '.join(stemmed_tokens)
    return(stemmedTweet)

#***************************************************************************


df_train = pd.read_csv(train_data, encoding='utf-8')
df_test = pd.read_csv(test_data, encoding='utf-8')

df_train['clean tweet'] = ""
df_test['clean tweet'] = ""

df_train_clean = pd.DataFrame(columns=['clean tweet', 'label'])
df_test_clean = pd.DataFrame(columns=['clean tweet', 'label'])


df_train = cleanData(df_train)
df_train_clean = df_train.copy().drop(columns=['tweet', 'id'])
df_train_clean = df_train_clean.reindex(columns= ['clean tweet', 'label'])
df_train_clean.to_csv(clean_train, sep=",")

'''
df_test = cleanData(df_test)
df_test_clean = df_test.copy().drop(columns=['tweet', 'id'])
df_test_clean = df_test_clean.reindex(columns= ['clean tweet', 'label'])
df_test_clean.to_csv(clean_test, sep=",")
'''

# Now tokenize and stem the cleaned up tweets
df_token = df_train_clean.copy()
df_token_test = df_test_clean.copy()
print('Calling tokenize')
df_token['tokens'] = df_token['clean tweet'].apply(nltk.word_tokenize)
df_token_test['tokens'] = df_token_test['clean tweet'].apply(nltk.word_tokenize)
print('tokenize done, calling stemmer')

# Calling stemmer
df_token['stemmed tweet'] = df_token['tokens'].apply(stemWords)
df_token_test['stemmed tweet'] = df_token_test['tokens'].apply(stemWords)

print('stemmer done, run sentiment analysis')

# Sentiment analysis with stemmed data
df_SA_stemmed = getSA(df_token)
df_SA_stemmed_test = getSA(df_token_test)
df_SA_stemmed.to_csv(sentiment_train_stemmed, sep=",")
print('SA on stemmed data ')
print(df_SA_stemmed.head())

# Sentiment analysis with unstemmed data
df_SA = getSA(df_train_clean)
df_SA_test = getSA(df_test_clean)
df_SA.to_csv(sentiment_train, sep=",")

print("******************TEST DATA *********************")
print(df_SA_test.head())
print("******************TEST DATA STEMMED *********************")
print(df_SA_stemmed_test.head())
print('SA on Unstemmed data ')
#print(df_SA.head())


#****************************************************************************

# Run Naive-Bayes classification on unstemmed data

#df_SA = pd.read_csv(sentiment_train, encoding='utf-8')
#df_test_clean = pd.read_csv(clean_test, encoding='utf-8')
# Using fit_transform, transform the corpus to a matrix.
count_vect = CountVectorizer()
train_df_counts = count_vect.fit_transform(df_SA['clean tweet'])

# Train a multinomial classifier using the training set using the features
# and the training set labels (getting 2 classifiers using label and sentiment based tagging
clf_label = MultinomialNB().fit(train_df_counts, df_SA['label'])
clf_senti = MultinomialNB().fit(train_df_counts, df_SA['sentiment'])

# Run Naive-Bayes classification on stemmed data

train_df_counts_stemmed = count_vect.fit_transform(df_SA_stemmed['stemmed tweet'])

# Train a multinomial classifier using the training set using the features
# and the training set labels (getting 2 classifiers using label and sentiment based tagging
clf_label_stemmed = MultinomialNB().fit(train_df_counts_stemmed, df_SA_stemmed['label'])
clf_senti_stemmed = MultinomialNB().fit(train_df_counts_stemmed, df_SA_stemmed['sentiment'])

# ***************************************************************************
# TEST - using clean and stemmed versions of the data on both classifiers
print('Get predictions for unstemmed test data')
test_SA = getPrediction(df_SA_test['clean tweet'])
test_SA_stemmed = getPrediction(df_SA_stemmed_test['stemmed tweet'])

print('Get predictions for Stemmed test data')
#print(test_SA)
test_SA.to_csv(prediction_test, sep=',')
test_SA_stemmed.to_csv(prediction_test_stemmed, sep=',')




