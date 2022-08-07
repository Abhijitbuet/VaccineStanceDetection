import string
from sklearn import metrics
import emoji as emoji
import nltk
import numpy as np
import pandas as pd
from keras.layers import LSTM
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import _stop_words
import matplotlib.pyplot as plt
import re
from sklearn.feature_selection import SelectKBest
from nltk.stem import PorterStemmer
from sklearn.metrics import multilabel_confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from wordcloud import WordCloud
from textblob import TextBlob
from nltk.corpus import words as nltk_words
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import scipy.sparse as sparse
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import validators
#import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
#nltk.download()
cachedStopWords = _stop_words.ENGLISH_STOP_WORDS
from sklearn import svm
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout, Embedding
from keras.utils import np_utils

i = 0
length = 0
total_alpha = 0
neutral_alpha = 0
negative_alpha = 0
positive_alpha = 0
total_word = 0
unique_words = set()
result_table = pd.DataFrame(columns=['Problem', 'fpr','tpr','auc'])
dictionary = dict.fromkeys(nltk_words.words(), None)
used_words = []


def textBlob_sentiment_score(x):
    return TextBlob(x).polarity


def is_english_word(word):
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False


def stem_word(word):
    ps = PorterStemmer()
    return ps.stem(word)


def check_valid_word(word):
    if word not in cachedStopWords:#is_english_word(word) and \
        return True
    else:
        return False


def remove_list_of_words(list, sentence):
    for word in list:
        sentence = sentence.replace(word,"")
    return sentence


def remove_uninformative_words(sentence):
    sentence = remove_list_of_words(['https' ],sentence)
    #love vs hate customized word removal

    return sentence


def convert_emoticons(sentence):
    sentence = sentence.replace(":'(", " crying ")
    sentence = sentence.replace("(y)", " thumbs up ")
    sentence = sentence.replace(":x", "")
    sentence = sentence.replace(":3", " goofy ")

    return sentence


def remove_valid_links(tweet):
    links = re.findall(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', tweet)
    for link in links:
        if validators.url(link):
            tweet = re.sub(link, ' ', tweet)
    return tweet


def sentence_preprocessor(sentence, i):
    global  length, total_alpha, neutral_alpha, negative_alpha, positive_alpha

    sentence = sentence.lower()
    sentence = convert_emoticons(sentence)
    sentence = emoji.demojize(sentence)
    total_alpha = total_alpha + sentence.count('@')
    length = length + len(sentence)
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = remove_valid_links(sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub('[0-9]+', '', sentence)

    new_sentence = ""
    words = sentence.split()
    for word in words:
        word = stem_word(word)
        if (check_valid_word(word)):
            new_sentence += word + " "
    new_sentence = remove_uninformative_words(new_sentence, i)
    new_sentence = new_sentence.strip()

    return new_sentence


def get_text_from_data(tweets):
    return ' '.join(tweets['content'])

'''
def create_wordcloud(tweet_dataframe):
    positive_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == 1]
    negative_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == -1]
    neutral_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == 0]

    positive_wordcloud = WordCloud().generate(get_text_from_data(positive_tweets))
    negative_wordcloud = WordCloud().generate(get_text_from_data(negative_tweets))
    neutral_wordcloud = WordCloud().generate(get_text_from_data(neutral_tweets))
    overall_wordcloud = WordCloud().generate(get_text_from_data(tweet_dataframe))

    plt.imshow(overall_wordcloud)
    plt.title("Wordcloud for All Classes")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.imshow(positive_wordcloud)
    plt.title("Positive Class Wordcloud")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.imshow(negative_wordcloud)
    plt.title("Negative Class Wordcloud")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.imshow(neutral_wordcloud)
    plt.title("Neutral Class Wordcloud")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
'''

def perform_preprocessing(tweet_data, i):
    return tweet_data.applymap(lambda s: sentence_preprocessor(s, i) if type(s) == str else s)


def run_statistics(row):
    global i, length, total_alpha, neutral_alpha, negative_alpha, positive_alpha, total_word, unique_words
    tweet = row['content']
    sentiment = row['sentiment']
    alpha_count = tweet.count(":(")
    if alpha_count > 0:
        total_alpha += alpha_count
        if sentiment == 0:
            neutral_alpha += alpha_count
        elif sentiment == 1:
            positive_alpha += alpha_count
        else:
            negative_alpha += alpha_count
    words = tweet.split()
    total_word += len(words)
    for word in words:
        unique_words.add(word)


def get_distribution_of_unique_words(tweet_dataframe, labels):
    positive_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == 1]
    negative_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == -1]
    neutral_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == 0]
    preprocessed_unique_words = set()
    positive_unique_words = set()
    negative_unique_words = set()
    neutral_unique_words = set()

    total_words_counts = [0, 0, 0, 0]

    for index, row in tweet_dataframe.iterrows():

        sentence = row["content"]
        sentiment = row["sentiment"]
        words_in_sentence = sentence.split()
        number_of_words = len(words_in_sentence)
        total_words_counts[0] += number_of_words
        if sentiment == 1:
            total_words_counts[1] += number_of_words
        elif sentiment == -1:
            total_words_counts[2] += number_of_words
        else:
            total_words_counts[3] += number_of_words
        for word in words_in_sentence:
            if sentiment == 1 and not (word in positive_unique_words):
                positive_unique_words.add(word)

            elif sentiment == -1 and not (word in negative_unique_words):
                negative_unique_words.add(word)

            else:

                if not (word in neutral_unique_words):
                    neutral_unique_words.add(word)
            preprocessed_unique_words.add(word)

    print("Total number of words in " + labels[0] + " class", (total_words_counts[1]))
    print("Total number words in " + labels[1] + " class", (total_words_counts[2]))
    print("Total number of neutral words: ", (total_words_counts[3]))
    print("Total number of total words: ", (total_words_counts[0]))

    print("Average number of words in " + labels[0] + " class", (total_words_counts[1]) / len(positive_tweets))
    print("Average number of  words: in " + labels[1] + " class", (total_words_counts[2]) / len(negative_tweets))
    print("Average number of neutral words: ", (total_words_counts[3]) / (len(neutral_tweets) + 0.00001))
    print("Average number of total words: ", (total_words_counts[0]) / len(tweet_dataframe))

    print("Total unique words after preprocessing: ", len(preprocessed_unique_words))
    print("Total unique words in " + labels[0] + " class: ", len(positive_unique_words))
    print("Total unique words in " + labels[1] + " class: ", len(negative_unique_words))
    print("Neutral unique words: ", len(neutral_unique_words))
    # print(preprocessed_unique_words)


def create_barchart_with_common_units(counter, count, title):
    labels = []
    values = []
    top_units = counter.most_common(count)
    for item in top_units:
        labels.append(item[0])
        values.append(item[1])
    plt.bar(labels, values, align='center', alpha=0.5)
    if (len(labels[0].split()) > 1):
        plt.xticks(labels, rotation=60)
        # plt.figure(figsize=(10,8))
    else:
        plt.xticks(labels)
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def get_top_ten_words(tweet_data, labels):
    preprocessed_unique_words = []
    positive_words = []
    negative_words = []
    neutral_words = []
    for index, row in tweet_data.iterrows():
        sentence = row["content"]
        sentiment = row["sentiment"]
        words_in_sentence = sentence.split()
        for word in words_in_sentence:
            preprocessed_unique_words.append(word)
            if sentiment == 1:
                positive_words.append(word)
            elif sentiment == -1:
                negative_words.append(word)
            else:
                neutral_words.append(word)
    positive_counter = collections.Counter(positive_words)
    negative_counter = collections.Counter(negative_words)
    neutral_counter = collections.Counter(neutral_words)
    print("Positive top 10: ", positive_counter.most_common(20))
    print("Negative top 10: ", negative_counter.most_common(20))
    #print("Neutral top 10: ", neutral_counter.most_common(20))
    #create_barchart_with_common_units(positive_counter, 10, "Top ten words for " + labels[0] + ' class')
    #create_barchart_with_common_units(negative_counter, 10, "Top ten words for " + labels[1] + ' class')
    if (len(labels) > 2):
        create_barchart_with_common_units(neutral_counter, 10, "Top ten words for " + labels[2] + ' class')


def sentiment_score_plotter(tweet_data, labels):
    positive_sentiment_scores = []
    negative_sentiment_scores = []
    neutral_sentiment_scores = []
    for index, row in tweet_data.iterrows():
        sentence = row[1]
        sentiment = row[0]

        sentiment_score_value = textBlob_sentiment_score(sentence)
        if sentiment == 1:
            positive_sentiment_scores.append(sentiment_score_value)
        elif sentiment == -1:
            negative_sentiment_scores.append(sentiment_score_value)
        elif sentiment == 0:
            neutral_sentiment_scores.append(sentiment_score_value)

    # print(labels[0]+" sentiment scores: ", positive_sentiment_scores)
    # print(labels[1]+" sentiment scores: ", negative_sentiment_scores)
    # print("Neutral sentiment scores: ", neutral_sentiment_scores)

    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

    plt.hist(positive_sentiment_scores, bins=100)
    plt.gca().set(title='Sentiment Scores for ' + labels[0], xlabel='Sentiment Scores', ylabel='Count')
    plt.show()

    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

    plt.hist(negative_sentiment_scores, bins=100)
    plt.gca().set(title='Sentiment Scores for ' + labels[1], xlabel='Sentiment Scores', ylabel='Count')
    plt.show()

    if (len(labels) > 2):
        plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
        plt.hist(neutral_sentiment_scores, bins=100)
        plt.gca().set(title='Neutral Sentiment Scores', ylabel='Count')
        plt.show()

    return positive_sentiment_scores, negative_sentiment_scores, neutral_sentiment_scores


def create_charts_with_rows(rows, sentiment):
    values = []
    chart_title = ""
    if sentiment == 1:
        chart_title = "Positive Sentiment TF-IDF Features"
    elif sentiment == -1:
        chart_title = "Negative Sentiment TF-IDF Features"
    else:
        chart_title = "Neutral Sentiment TF-IDF Features"
    i = 0

    for row in rows:
        X, Y, Z = sparse.find(row)

        for value in Z:
            values.append(value)
        i += 1
        if i % 10000 == 0:
            print(i)
    # print(values)
    rounded_values = [round(elem, 2) for elem in values]
    # print(rounded_values)
    # plt.xticks(np.arange(0, 1, step=0.2))
    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    plt.hist(rounded_values, bins=20)
    plt.gca().set(title=chart_title, ylabel='Count', xlabel='TF-IDF Value')
    plt.show()


def run_statisics_of_tf_idf(unigram_vectorizer, X, tweet_dataframe, ngram, labels):
    positive_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == 1]
    negative_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == -1]
    neutral_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == 0]
    features = unigram_vectorizer.get_feature_names()
    print("Size of corpus: ", len(features))
    positive_rows = []
    negative_rows = []
    neutral_rows = []

    total_ngram_count = 0
    positive_ngram_count = 0
    negative_ngram_count = 0
    neutral_ngram_count = 0

    positive_common_words = []
    negative_common_words = []
    neutral_common_words = []
    total_common_words = []
    i = 0
    for index, row in tweet_dataframe.iterrows():
        # print(i)
        sentiment = row["sentiment"]
        row = X[i]
        if sentiment == 1:
            positive_rows.append(row)
            for index in row.indices:
                positive_word = features[index]
                positive_common_words.append(positive_word)
            positive_ngram_count += len(row.indices)
        elif sentiment == -1:
            negative_rows.append(row)
            for index in row.indices:
                negative_word = features[index]
                negative_common_words.append(negative_word)
            negative_ngram_count += len(row.indices)
        else:
            neutral_rows.append(row)
            for index in row.indices:
                neutral_word = features[index]
                neutral_common_words.append(neutral_word)
            neutral_ngram_count += len(row.indices)
        for index in row.indices:
            word = features[index]
            total_common_words.append(word)
        total_ngram_count += len(row.indices)
        i += 1

    total_counter = collections.Counter(total_common_words)
    positive_counter = collections.Counter(positive_common_words)
    negative_counter = collections.Counter(negative_common_words)
    neutral_counter = collections.Counter(neutral_common_words)
    print("Size of " + labels[0] + " class " + ngram + "s: ", len(positive_counter))
    print("Size of " + labels[1] + " class " + ngram + "s: ", len(negative_counter))
    print("Size of neutral class " + ngram + "s: ", len(neutral_counter))

    print("Average " + ngram + " count in " + labels[0] + " class: ", positive_ngram_count * 1.0 / len(positive_tweets))
    print("Average " + ngram + " count in " + labels[1] + " class: ", negative_ngram_count * 1.0 / len(negative_tweets))
    print("Average " + ngram + " count in neutral class: ", neutral_ngram_count * 1.0 / (len(neutral_tweets) + 0.0001))
    print("Average " + ngram + " count in total: ", total_ngram_count * 1.0 / len(tweet_dataframe))

    positive_most_common_words = positive_counter.most_common(10)
    print("Positive common " + ngram + "s: ", positive_most_common_words)
    negative_most_common_words = negative_counter.most_common(10)
    print("Negative common " + ngram + "s: ", negative_most_common_words)
    neutral_most_common_words = neutral_counter.most_common(10)
    print("Neutral common " + ngram + "s: ", neutral_most_common_words)

    print("Total common " + ngram + "s: ", total_counter.most_common(10))
    # print(positive_columns)
    create_barchart_with_common_units(positive_counter, 10, "Top ten " + ngram + "s for " + labels[0] + ' class')
    create_barchart_with_common_units(negative_counter, 10, "Top ten " + ngram + "s for " + labels[1] + ' class')
    if (len(labels) > 2):
        create_barchart_with_common_units(neutral_counter, 10, "Top ten " + ngram + "s for " + labels[2] + ' class')


def tf_idf_vectorize(tweet_dataframe, labels):
    corpus = tweet_dataframe.loc[:, "content"]
    Y = tweet_dataframe.loc[:, 'sentiment']

    unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    unigram_tf_idf = unigram_vectorizer.fit_transform(corpus)
    # run_statisics_of_tf_idf(unigram_vectorizer, unigram_tf_idf, tweet_dataframe, 'unigram', labels)

    #print("\n\n")
    #bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    # bigram_tf_idf = bigram_vectorizer.fit_transform(corpus)
    # run_statisics_of_tf_idf(bigram_vectorizer, bigram_tf_idf, tweet_dataframe, 'bigram', labels)

    #print("\n\n")
    #unigram_and_bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    # unigram_and_bigram_tf_idf = unigram_and_bigram_vectorizer.fit_transform(corpus)
    # run_statisics_of_tf_idf(unigram_and_bigram_vectorizer, unigram_and_bigram_tf_idf, tweet_dataframe,
    #                        'unigram and bigram', labels)

    return unigram_tf_idf, Y


def create_barchart(labels, values, title):
    plt.bar(labels, values, align='center', alpha=0.5)
    plt.xticks(labels)
    plt.ylabel('count')
    plt.title(title)

    plt.show()


def get_class_counts(tweet_dataframe, labels):
    positive_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == 1]
    negative_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == -1]
    values = [len(positive_tweets), len(negative_tweets)]
    if (len(labels) > 2):
        values.append(len(tweet_dataframe.loc[tweet_dataframe['sentiment'] == 0]))
    return values

''' 
def create_wordcloud_for_binary(tweet_dataframe, labels):
    positive_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == 1]
    negative_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == -1]
    all_tweets = pd.concat([positive_tweets, negative_tweets])

    positive_wordcloud = WordCloud().generate(get_text_from_data(positive_tweets))
    negative_wordcloud = WordCloud().generate(get_text_from_data(negative_tweets))

    overall_wordcloud = WordCloud().generate(get_text_from_data(all_tweets))

    plt.imshow(overall_wordcloud)
    plt.title("Overall Wordcloud")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.imshow(positive_wordcloud)
    plt.title(labels[0])
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.imshow(negative_wordcloud)
    plt.title(labels[1])
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
'''

def calculate_binary_accuracies_of_textBlob(positive_scores, negative_scores, neutral_scores, labels, threshold=0):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_class = []
    predicted_class = []
    for value in positive_scores:
        true_class.append(labels[0])
        if value >= threshold:
            true_positive += 1
            predicted_class.append(labels[0])
        else:
            false_negative += 1
            predicted_class.append(labels[1])
    for value in negative_scores:
        true_class.append(labels[1])
        if value >= threshold:
            true_positive += 1
            predicted_class.append(labels[0])
        else:
            false_negative += 1
            predicted_class.append(labels[1])
    print("Accuracy of TextBlob for " + labels[0] + " class: ",
          true_positive / (true_positive + false_negative + .00001))
    print("Accuracy of TextBlob for " + labels[1] + " class: ",
          true_negative / (true_negative + false_positive + .00001))

    multilabel_confusion_matrix = confusion_matrix(true_class, predicted_class,
                                                   labels=labels)
    print(multilabel_confusion_matrix)

    fig, ax = plot_confusion_matrix(conf_mat=multilabel_confusion_matrix,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    )
    plt.xticks(np.arange(0, 2), labels)

    ax.set(yticks=[-.5, 0, 0.5, 1, 1.5],

           yticklabels=['', labels[0], '', labels[1]])

    plt.show()

    # create_barchart(['true positive','false negative', 'true negative', 'false positive'], [true_positive, false_negative,true_negative, false_positive], "Performance of TextBlob in "+labels[0]
    #                +" vs "+labels[1]+ " classification")


def get_original_labels(true_class, predicted_class, labels):
    y_true = []
    y_pred = []
    for value in true_class:
        if value == 1:
            y_true.append(labels[0])
        else:
            y_true.append(labels[1])
    for value in predicted_class:
        if value == 1:
            y_pred.append(labels[0])
        else:
            y_pred.append(labels[1])
    return y_true, y_pred


def plot_binary_confusion_matrix(true_class, predicted_class, labels):
    y_true, y_pred = get_original_labels(true_class, predicted_class, labels)
    multilabel_confusion_matrix = confusion_matrix(y_true, y_pred,
                                                   labels=labels)
    print(multilabel_confusion_matrix)

    fig, ax = plot_confusion_matrix(conf_mat=multilabel_confusion_matrix,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    )
    plt.xticks(np.arange(0, 2), labels)

    ax.set(yticks=[-.5, 0, 0.5, 1, 1.5],

           yticklabels=['', labels[0], '', labels[1]])

    plt.show()


def calculate_multiclass_accuracies_of_textBlob(positive_scores, negative_scores, neutral_scores, labels):
    predicted_class = []
    true_class = []
    for value in positive_scores:
        true_class.append("Positive")
        if value >= 0.2:
            predicted_class.append("Positive")
        elif value < 0.2 and value >= -0.2:
            predicted_class.append("Neutral")
        else:
            predicted_class.append("Negative")

    for value in negative_scores:
        true_class.append("Negative")
        if value >= 0.2:
            predicted_class.append("Positive")
        elif value < 0.2 and value >= -0.2:
            predicted_class.append("Neutral")
        else:
            predicted_class.append("Negative")

    for value in neutral_scores:
        true_class.append("Neutral")
        if value >= 0.2:
            predicted_class.append("Positive")
        elif value < 0.2 and value >= -0.2:
            predicted_class.append("Neutral")
        else:
            predicted_class.append("Negative")

    # plot_binary_confusion_matrix()
    multilabel_confusion_matrix = confusion_matrix(true_class, predicted_class,
                                                   labels=["Positive", "Negative", "Neutral"])
    print(multilabel_confusion_matrix)

    fig, ax = plot_confusion_matrix(conf_mat=multilabel_confusion_matrix,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    )
    plt.xticks(np.arange(0, 3), labels)

    ax.set(yticks=[-.5, 0, 0.5, 1, 1.5, 2, 2.5],

           yticklabels=['', "Positive", '', 'Negative', '', 'Neutral'])

    plt.show()
    # exit(1)


def vader_sentiment_score(tweet):
    return SentimentIntensityAnalyzer().polarity_scores(tweet)['compound']


def get_all_unique_words(tweet_data):
    all_unique_words = set()
    for index, row in tweet_data.iterrows():
        sentence = row['content']
        words = sentence.split()
        for word in words:
            if word not in all_unique_words:
                all_unique_words.add(word)
    return all_unique_words


def get_unique_word_count( words):
    return len(set(words))


def get_stop_word_count(tweet):
    count = 0
    words = tweet.split()
    for word in words:
        if word in cachedStopWords:
            count += 1
    return count


def get_number_of_all_capitals(words):
    count = 0

    for word in words:
        if word.isupper():
            count += 1

    return count


def get_number_of_youtube_links(links):
    count = 0

    for link in links:
        if youtube_url_validation(link):
            count += 1
            # print('youtube found')
    return count


def get_number_of_image_links(links):
    count = 0
    r_image = re.compile(r".*\.(jpg|png|gif|jpeg|JPG|JPEG|PNG|GIF)$")

    for link in links:

        if validators.url(link):
            # print(link)
            if r_image.match(link) or link.find('twitpic') > 0:
                # print("Valid link: "+link)
                count += 1

    return count


def is_ellipse_ending(tweet):
    if tweet.endswith('...'):
        return 1
    else:
        return 0


def plot_average(values, Y, labels, title):

    positive_count = 0
    negative_count = 0
    positive_total = 0
    negative_total = 0
    data = {'Values': values,
            'Y': Y}
    dataframe = pd.DataFrame(data)
    for index,row in dataframe.iterrows():
        if row['Y']==1:
            positive_count+=1
            positive_total+= row['Values']
        else:
            negative_count+=1
            negative_total+=row['Values']
    positive_average = positive_total/positive_count
    negative_average = negative_total/negative_count
    print(title+ ": "+ labels[0]+ ": "+str(round(positive_average,2))+", "+labels[1]+": "+str(round(negative_average,2)))


def calculate_emoticons_in_tweet(tweet):
    emoticons = [":)", ":'(", ":p", ":D",":d", ":x", ":Y", ":y", " :/", ":3 ", ":(", ":X", ":')", ":P", ";)"]
    count = 0
    for emoticon in emoticons:
        count+= tweet.count(emoticon)

    return count


def getMetaData(tweet_data, labels, i):
    Y = tweet_data.loc[:, 'sentiment']
    textBlob_scores = []
    vader_scores = []
    number_of_words = []
    number_of_links = []
    number_of_characters = []
    number_of_emoticons = []
    number_of_unique_words = []
    number_of_stopwords = []
    number_of_all_capitals = []
    number_of_at_counts = []
    number_of_youtube_links = []
    number_of_picture_links = []
    ellipse_endings = []
    exclamatory_counts = []
    hashtag_counts = []
    punctuation_counts = []
    mention_counts = []
    #all_unique_words = get_all_unique_words(tweet_data)
    Y = tweet_data.loc[:, 'sentiment']
    for index, row in tweet_data.iterrows():
        tweet = row['content']
        links = re.findall(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', tweet)
        textBlob_score = textBlob_sentiment_score(tweet)
        #textBlob_scores.append(textBlob_score)
        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        punctuation_count = count(tweet, string.punctuation)
        punctuation_counts.append(punctuation_count)

        if textBlob_score>=0:
            textBlob_scores.append(1)
        else:
            textBlob_scores.append(0)


        vader_score = vader_sentiment_score(tweet)
        #vader_scores.append(vader_score)


        if vader_score >= 0:
            vader_scores.append(1)
        else:
            vader_scores.append(0)

        words = tweet.split()
        hashtag_count = 0
        mention_count = 0
        for word in words:
            if word.startswith('#'):
                hashtag_count += 1
            elif word.startswith('@'):
                mention_count+=1

        number_of_words.append(len(words))

        number_of_characters.append(len(tweet))

        number_of_links.append(len(links))

        number_of_emoticons.append(calculate_emoticons_in_tweet(tweet))

        number_of_unique_word_in_tweet = get_unique_word_count( words)
        number_of_unique_words.append(number_of_unique_word_in_tweet)

        number_of_stopwords.append(get_stop_word_count(tweet))

        number_of_all_capitals_in_tweet = get_number_of_all_capitals(words)
        number_of_all_capitals.append(number_of_all_capitals_in_tweet)

        number_of_at_counts.append(tweet.count('@'))

        number_of_youtube_links_in_tweet = get_number_of_youtube_links(links)
        number_of_youtube_links.append(number_of_youtube_links_in_tweet)

        number_of_image_links_in_tweet = get_number_of_image_links(links)
        number_of_picture_links.append(number_of_image_links_in_tweet)

        ellipse_ending = is_ellipse_ending(tweet)
        ellipse_endings.append(ellipse_ending)

        exclamatory_counts.append(tweet.count("!"))
        hashtag_counts.append(hashtag_count)
        mention_counts.append(mention_count)

    '''
    plot_average(number_of_words, Y, labels, "Average total words per class")
    plot_average(number_of_unique_words, Y, labels, "Average unique words per tweet per class")
    plot_average(textBlob_scores, Y, labels, "Average textblob score per tweet per class")
    plot_average(vader_scores, Y, labels, "Average vader score per tweet per class")
    plot_average(number_of_links, Y, labels, "Average links per tweet per class")
    plot_average(number_of_emoticons, Y, labels, "Average emoticons per tweet per class")
    plot_average(number_of_characters, Y, labels, "Average number of characters per tweet per class")
    plot_average(number_of_stopwords, Y, labels, "Average stopwords per tweet per class")
    plot_average(number_of_all_capitals, Y, labels, "Average all capitals per tweet per class")

    plot_average(number_of_at_counts, Y, labels, "Average at counts per tweet per class")
    plot_average(number_of_youtube_links, Y, labels, "Average youtube links per tweet per class")
    plot_average(number_of_picture_links, Y, labels, "Average image links per tweet per class")
    plot_average(ellipse_endings, Y, labels, "Average ellipse endings per tweet per class")
    '''
    #plot_average(hashtag_counts, Y, labels, "Average Hashtage counts per tweet per class")
    #plot_average(exclamatory_counts, Y, labels, "Average exclamatory counts per tweet per class")

    data = {'Number of words': number_of_words,
            'textBlob scores': textBlob_scores,
            'vader scores': vader_scores,
            #'Number of links': number_of_links,
            'Number of characters': number_of_characters,
             'Number of stopwords': number_of_stopwords,
            'Number of all capitals': number_of_all_capitals,
            'Number of at counts': number_of_at_counts,
           # 'Ellipse Ending': ellipse_endings,
            'Number of exclamatory signs': exclamatory_counts,
            #'Number of picture links': number_of_picture_links,
           #'Hashtag counts': hashtag_counts,
            'Unique word count': number_of_unique_words,
          #  'Emoticon counts': number_of_emoticons,
            #'Youtube link count': number_of_youtube_links,
          #  'Punctuation count': punctuation_counts,
            'Mention count': mention_counts

            }

    if i==1:
        data = {#'Number of words': number_of_words,
                'textBlob scores': textBlob_scores,
                'vader scores': vader_scores,
                # 'Number of links': number_of_links,
                'Number of characters': number_of_characters,
                'Number of stopwords': number_of_stopwords,
                'Number of all capitals': number_of_all_capitals,
                'Number of at counts': number_of_at_counts,
                'Ellipse Ending': ellipse_endings,
                'Number of exclamatory signs': exclamatory_counts,
                # 'Number of picture links': number_of_picture_links,
                # 'Hashtag counts': hashtag_counts,
               # 'Unique word count': number_of_unique_words,
                #  'Emoticon counts': number_of_emoticons,
                # 'Youtube link count': number_of_youtube_links,
                  'Punctuation count': punctuation_counts,
                'Mention count': mention_counts

                }
    elif i == 2:
            data = {   'Number of words': number_of_words,
                'textBlob scores': textBlob_scores,
                'vader scores': vader_scores,
                # 'Number of links': number_of_links,
                'Number of characters': number_of_characters,
                'Number of stopwords': number_of_stopwords,
                'Number of all capitals': number_of_all_capitals,
                #'Number of at counts': number_of_at_counts,
                #'Ellipse Ending': ellipse_endings,
                'Number of exclamatory signs': exclamatory_counts,
                # 'Number of picture links': number_of_picture_links,
                 'Hashtag counts': hashtag_counts,
                'Unique word count': number_of_unique_words,
                #  'Emoticon counts': number_of_emoticons,
                # 'Youtube link count': number_of_youtube_links,
                'Punctuation count': punctuation_counts,
                #'Mention count': mention_counts

            }
    elif i==3 :
        data = {'Number of words': number_of_words,
                'textBlob scores': textBlob_scores,
                'vader scores': vader_scores,
                # 'Number of links': number_of_links,
                'Number of characters': number_of_characters,
                'Number of stopwords': number_of_stopwords,
                #'Number of all capitals': number_of_all_capitals,
                'Number of at counts': number_of_at_counts,
               # 'Ellipse Ending': ellipse_endings,
                'Number of exclamatory signs': exclamatory_counts,
                # 'Number of picture links': number_of_picture_links,
                # 'Hashtag counts': hashtag_counts,
                'Unique word count': number_of_unique_words,
                #  'Emoticon counts': number_of_emoticons,
                # 'Youtube link count': number_of_youtube_links,
                  'Punctuation count': punctuation_counts,
                'Mention count': mention_counts

                }

    df = pd.DataFrame(data)
    ''' 
    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=17)
    fit = bestfeatures.fit(df, Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(df.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Feature', 'Score']  # naming the dataframe columns
    
    print(featureScores.nlargest(17, 'Score'))
    '''
    return df


def youtube_url_validation(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    youtube_regex_match = re.match(youtube_regex, url)
    if youtube_regex_match:
        return youtube_regex_match

    return youtube_regex_match


def get_tweet_data_from_problem(all_data, i):
    tweet_data = None
    labels = None
    if i==5:
        print("Positive vs Negative vs Neutral")
        sentiment_mapper = {"sentiment": {"worry": -1, "boredom": -1, "sadness":-1, "anger": -1, "hate": -1, "love": 1,
                                          "surprise": 1, "happiness": 1, "relief": 1, "fun":1, "enthusiasm": 1,
                                          "neutral": 0,
                                          "empty": 2}}
        labels = ['Positive', 'Negative', 'Neutral']
        all_data.replace(sentiment_mapper, inplace=True)
        positive_tweets = all_data.loc[all_data['sentiment'] == 1]
        negative_tweets = all_data.loc[all_data['sentiment'] == -1]
        neutral_tweets = all_data.loc[all_data['sentiment'] == 0]
        tweet_data = pd.concat([positive_tweets, negative_tweets, neutral_tweets])
    else:
        print(i)
        if i == 0:
            print("Love vs Hate")
            sentiment_mapper = {"sentiment": {"worry": 2, "boredom": 2, "sadness": 2, "anger": 2, "hate": 0, "love": 1,
                                              "surprise": 2, "happiness":2, "relief": 2, "fun": 2, "enthusiasm": 2,
                                              "neutral": 2,
                                              "empty": 2}}
            labels = ['Love', 'Hate']


        elif i == 1:
            print("Joy vs Sadness")
            sentiment_mapper = {"sentiment": {"worry": 2, "boredom": 2, "sadness": 0, "anger": 2, "hate": 2, "love": 2,
                                              "surprise": 2, "happiness": 1, "relief": 1, "fun": 1, "enthusiasm": 2,
                                              "neutral": 2,
                                              "empty": 2}}
            labels = ['Joy', 'Sadness']

        elif i == 2:
            print("Interest vs Surprise")

            sentiment_mapper = {"sentiment": {"worry": 2, "boredom": 2, "sadness": 2, "anger": 2, "hate": 2, "love": 1,
                                              "surprise": 0, "happiness": 2, "relief": 2, "fun": 2, "enthusiasm": 1,
                                              "neutral": 2,
                                              "empty": 2}}
            labels = ['Interest', 'Surprise']

        elif i == 3:
            print("Trust vs Disgust")

            sentiment_mapper = {"sentiment": {"worry": 0, "boredom": 0, "sadness": 2, "anger": 2, "hate": 2, "love": 1,
                                              "surprise": 2, "happiness": 2, "relief": 1, "fun": 2, "enthusiasm": 2,
                                              "neutral": 2,
                                              "empty": 2}}
            labels = ['Trust', 'Disgust']

        elif i == 4:
            print("Fear vs Anger")

            sentiment_mapper = {"sentiment": {"worry": 1, "boredom": 2, "sadness": 2, "anger": 0, "hate": 0, "love": 2,
                                              "surprise": 2, "happiness": 2, "relief":2, "fun": 2, "enthusiasm": 2,
                                              "neutral": 2,
                                              "empty": 2}}
            labels = ['Fear', 'Anger']

        all_data.replace(sentiment_mapper, inplace=True)
        positive_tweets = all_data.loc[all_data['sentiment'] == 1]
        negative_tweets = all_data.loc[all_data['sentiment'] == 0]
        tweet_data = pd.concat([positive_tweets, negative_tweets])
    return tweet_data,labels


def run_steps_upto_feature_extraction(i):
    all_data = pd.read_csv('text_emotion.csv')
    # positive_tweets = all_data.loc[all_data['sentiment'] == 1]
    # negative_tweets = all_data.loc[all_data['sentiment'] == -1]
    #
    tweet_data = None
    labels = None

    tweet_data, labels = get_tweet_data_from_problem(all_data, i )

    #neutral_tweets = all_data.loc[all_data['sentiment'] == 0]

    # positive_scores, negative_scores, neutral_scores = sentiment_score_plotter(tweet_data, labels)
    # calculate_binary_accuracies_of_textBlob(positive_scores, negative_scores, neutral_scores, labels, threshold=0.1)
    # calculate_multiclass_accuracies_of_textBlob(positive_scores, negative_scores, neutral_scores, labels)
    # for index, row in tweet_data.iterrows():
    # run_statistics(row)
    # break

    #y = get_class_counts(tweet_data, labels)
    # create_barchart(labels, y, 'count')

    print("\n=================Before preprocessing================")
    # get_distribution_of_unique_words(tweet_data, labels)

    metadata = getMetaData(tweet_data, labels, i)

    #metadata.to_csv(path_or_buf='metadata.csv')
    tweet_data = perform_preprocessing(tweet_data, i)
    print(negative_alpha)
    print(positive_alpha)
    print(neutral_alpha)

    #print("Average characters in each tweet: ", length * 1.0 / len(tweet_data))
    #print("Average words in each tweet: ", total_word * 1.0 / len(tweet_data))
    #print("Total unique words: ", len(unique_words))
    print("\n=================After preprocessing================")
    # get_distribution_of_unique_words(tweet_data, labels)

    # positive_scores, negative_scores, neutral_scores = sentiment_score_plotter(tweet_data, labels)
    # calculate_binary_accuracies_of_textBlob(positive_scores, negative_scores, neutral_scores, labels, threshold=0.1)

    #get_top_ten_words(tweet_data, labels)
    # create_wordcloud(tweet_data)
    #create_wordcloud_for_binary(tweet_data,labels)
    X, Y = tf_idf_vectorize(tweet_data, labels)
    X = X.todense()
    #print(X)
    return X, Y, metadata, labels
def  get_positive_percentage(Y):
    count = 0
    for value in Y:
        if value == 1:
            count+=1
    percentage = count/len(Y)
    return percentage


def convert_y_pred(y_pred):
    y=[]
    for value in y_pred:
        if value==True:
            y.append(1)
        else: y.append(-1)
    return y


def get_problem_name(i):
    if i==0:
        return "Love vs Hate"
    elif i==1:
        return "Joy vs Sadness"
    elif i==2:
        return "Anticipation vs Surprise"
    else: 
        return "Trust vs Disgust"

def run_neural_network(X_train, y_train, X_test, y_test):
    # Neural network
    print(y_train)
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)
    model = Sequential()
    model.add(Dense(30,  activation='relu' , input_dim=len(X_train.columns)))
    model.add(Dropout(0.2))
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Model Training

    model.fit(X_train, y_train, epochs=100, verbose=1)


    # evaluate the keras model
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))

    #print(model.predict(X_test))
    #print(y_test)


def run_LSTM(X_train, y_train, X_test, y_test):
    # Neural network
    #print(y_train)
    #y_train = np_utils.to_categorical(y_train, 2)
    #y_test = np_utils.to_categorical(y_test, 2)
    X_train = X_train.to_numpy()[:,:,None]
    #X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, X_valid.shape[1]))
    X_test = X_test.to_numpy()[:,:,None]
    model = Sequential()
    model.add(Embedding(2500, 64, input_length=X_train.shape[1]))
    model.add(LSTM(32))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Model Training

    model.fit(X_train, y_train, epochs=10, verbose=1)

    # evaluate the keras model
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))

    #print(model.predict(X_test))
    #print(y_test)


def run_supervised_ML(all_data, Y, labels):
    # for degrees in range (2,5,1):
    # print ("trees: "+str(trees))

    X_train, X_test, y_train, y_test = train_test_split(
        all_data, Y, test_size=0.25, random_state=42)
    
    #run_neural_network(X_train, y_train, X_test, y_test)
    run_LSTM(X_train, y_train, X_test, y_test)
    #clf = MultinomialNB()
    #clf = svm.SVC(kernel='linear')
    #clf = RandomForestClassifier(n_estimators=100)
    '''
    print("MLP")
    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(10, 5, 2), max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print(clf.feature_importances_[-13:])
    # y_probas = clf.predict_proba(X_test)
    # skplt.metrics.plot_roc_curve(y_test, y_probas,curves='macro'  )
    # plt.show()

    # plot_binary_confusion_matrix(y_test, y_pred, labels)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
  
    print("Recall/Sensitivity : " + str(round(tp / (tp + fn), 2)))
    print("Specificity : " + str(round(tn / (tn + fp), 2)))
    y_proba = clf.predict_proba(X_test)[::, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba, pos_label=1)
    print("AUC of MLP: " + str(metrics.auc(fpr, tpr)))

    print("\nSVM")
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print(clf.feature_importances_[-13:])
    # y_probas = clf.predict_proba(X_test)
    # skplt.metrics.plot_roc_curve(y_test, y_probas,curves='macro'  )
    # plt.show()

    # plot_binary_confusion_matrix(y_test, y_pred, labels)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
    print("Recall/Sensitivity : " + str(round(tp / (tp + fn), 2)))
    print("Specificity : " + str(round(tn / (tn + fp), 2)))
    y_proba = clf.predict_proba(X_test)[::, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba, pos_label=1)
    print("AUC of SVM: " + str(metrics.auc(fpr, tpr)))
    
    positive_class_percentage = get_positive_percentage(Y)
    '''
    ''' 
    print("\nRF")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    #y_pred = (clf.predict_proba(X_test)[:,1] >= positive_class_percentage).astype(bool)
    #y_pred = convert_y_pred(y_pred)

    #print(y_pred)
    print("Feature Importances: ")
    print(clf.feature_importances_[-10:])
    y_probas = clf.predict_proba(X_test)[::,1]




    y_pred =clf.predict(X_test)# (y_probas[:,1] >= 0.5+(positive_class_percentage-0.5)/2).astype(bool)
    y_pred = convert_y_pred(y_pred)
    #plot_binary_confusion_matrix(y_test, y_pred, labels)

    #skplt.metrics.plot_roc_curve(y_test, y_probas, curves='micro')
    #plt.show()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
    print("Recall/Sensitivity : " + str(round(tp/(tp+fn), 2)))
    print("Specificity : " + str(round(tn / (tn + fp), 2)))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probas)
    #print("FPR"+str(fpr))
    #print("TPR" + str(tpr))
    print("AUC: "+str(metrics.auc(fpr, tpr)))
    global result_table
    result_table = result_table.append({'Problem': get_problem_name(i),
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': metrics.auc(fpr, tpr)}, ignore_index=True)
    '''
    '''
    print("\nNB")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print(clf.feature_importances_[-13:])
    y_probas = clf.predict_proba(X_test)[::,1]
    # skplt.metrics.plot_roc_curve(y_test, y_probas,curves='macro'  )
    # plt.show()

    # plot_binary_confusion_matrix(y_test, y_pred, labels)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
    print("Recall/Sensitivity : " + str(round(tp / (tp + fn), 2)))
    print("Specificity : " + str(round(tn / (tn + fp), 2)))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probas, pos_label=1)
    print("AUC of NB: " + str(metrics.auc(fpr, tpr)))
    

    print("\n Gradient Boosting")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Accuracy: "+str(round((tp+tn) / (tn + fp+tp+fn), 2)))
    print("Recall/Sensitivity : " + str(round(tp / (tp + fn), 2)))
    print("Specificity : " + str(round(tn / (tn + fp), 2)))
    y_proba = clf.predict_proba(X_test)[::, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba, pos_label=1)
    print("AUC: " + str(metrics.auc(fpr, tpr)))
    '''







def getModified_Y(Y):
    y = []
    for value in Y:
        if value ==1:
            y.append(0)
        elif value == -1:
            y.append(1)
        else:
            y.append(2)
    return y



def run_supervised_ML_CV(all_data, Y, labels):


    print("MLP")
    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(10, 5, 2))
    cv_scores = cross_val_score(clf, all_data, Y, cv=5)


    print("cross validation Accuracy : " + str(round(np.mean(cv_scores), 2)))


    print("\nSVM")
    clf = svm.SVC(kernel='linear', probability=True)

    cv_scores = cross_val_score(clf, all_data, Y, cv=5)

    print("cross validation Accuracy : " + str(round(np.mean(cv_scores), 2)))

    print("\nRF")
    clf = RandomForestClassifier( n_estimators=100)
    cv_scores = cross_val_score(clf, all_data, Y, cv=5)

    print("cross validation Accuracy : " + str(round(np.mean(cv_scores), 2)))


def plot_combined_ROC():
    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label=result_table.loc[i]['Problem']+", AUC= "+str(round(result_table.loc[i]['auc'],2)))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()


''' 
for i in range (0,4):
    X, Y, meta_data, labels = run_steps_upto_feature_extraction(i)
    tf_idf_frame = pd.DataFrame(X)
    combined_data = pd.concat([ tf_idf_frame, meta_data], axis=1)
    # print(combined_data)
    run_supervised_ML(combined_data, Y, labels)

    #run_unsupervised_ML(combined_data, Y, labels)
#plot_combined_ROC()
#print(combined_data)
'''