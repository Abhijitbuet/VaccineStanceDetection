import emoji
import re
import pandas as panda
import string
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob
from nltk.corpus import words as nltk_words
from SentimentAnalyzer import get_stop_word_count, get_unique_word_count, stem_word, get_original_labels, \
    remove_valid_links, remove_uninformative_words
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import auc
from keras.models import Sequential
from keras.layers import Dense,Dropout, Embedding
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from wordcloud import WordCloud
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeClassifierCV
import tensorflow
import nltk
#nltk.download()
# create a hashed dictionary of all  english words in NLTK
dictionary = dict.fromkeys(nltk_words.words(), None)

# create a cached stop words list to speed up
cachedStopWords = stop_words.ENGLISH_STOP_WORDS


# if the word is not a stop word and it is in NLTK English dictionary
def is_English_word(word):
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False


# calculate textBlob sentiment score  of a tweet
def textBlob_sentiment_score(x):
    return TextBlob(x).polarity


# calculate vader sentiment score of a tweet
def vader_sentiment_score(tweet):
    return SentimentIntensityAnalyzer().polarity_scores(tweet)['compound']

def create_wordcloud(dataset):
    positive_tweets = dataset.loc[dataset['sentiment'] == 1]
    negative_tweets = dataset.loc[dataset['sentiment'] == -1]
    print(len(positive_tweets))
    #neutral_tweets = tweet_dataframe.loc[tweet_dataframe['sentiment'] == 0]

    positive_wordcloud = WordCloud().generate(get_text_from_data(positive_tweets))
    print(positive_wordcloud.words_.keys())
    negative_wordcloud = WordCloud().generate(get_text_from_data(negative_tweets))
    #neutral_wordcloud = WordCloud().generate(get_text_from_data(neutral_tweets))
    overall_wordcloud = WordCloud().generate(get_text_from_data(dataset))

    plt.imshow(overall_wordcloud)
    plt.title("Wordcloud for All Classes")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.imshow(positive_wordcloud)
    plt.title("Pro-vaccine Wordcloud")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.imshow(negative_wordcloud)
    plt.title("Antivaccine Wordcloud")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
def check_valid_word(word):
    if word not in cachedStopWords:  # is_english_word(word) and \
        return True
    else:
        return False


# collect all the meadata
def getMetaData(tweet_data):
    # create empty lists for all available metadata in this dataset
    textBlob_scores = []
    vader_scores = []
    number_of_words = []
    number_of_characters = []
    number_of_unique_words = []
    number_of_stopwords = []

    for index, row in tweet_data.iterrows():
        # get text
        tweet = row['text']
        # calculate textblob sentiment score
        textBlob_score = textBlob_sentiment_score(tweet)
        textBlob_scores.append(textBlob_score)
        # calculate vader sentiment score
        vader_score = vader_sentiment_score(tweet)
        vader_scores.append(vader_score)
        # get words of the tweet
        words = tweet.split()
        # count number of words and append it to a list
        number_of_words.append(len(words))
        # count number of characters and append it to a list
        number_of_characters.append(len(tweet))
        # count number of unique words and append it to a list
        number_of_unique_word_in_tweet = get_unique_word_count(words)
        number_of_unique_words.append(number_of_unique_word_in_tweet)
        # count number of stopwords and append it to a list
        number_of_stopwords.append(get_stop_word_count(tweet))

    # prepare key and values list for creating metadata frame
    data = {'Number of words': number_of_words,
            'textBlob scores': textBlob_scores,
            'vader scores': vader_scores,
            'Number of characters': number_of_characters,
            'Number of stopwords': number_of_stopwords,
            'Unique word count': number_of_unique_words,
            }

    # create data frame of metadata
    metadata = panda.DataFrame(data)

    return metadata


# get dataset using pandas library
def get_dataset(classification_type):
    dataset = panda.read_csv('tweet_outputs/labeled_tweets.txt',sep=';separator;',engine='python')
    #create_wordcloud(dataset)
    positive_tweets = dataset.loc[dataset['sentiment'] == 1]
    negative_tweets = dataset.loc[dataset['sentiment'] == -1]
    selected_dataset = panda.concat([positive_tweets, negative_tweets])
    print("Total dataset size: "+str(len(dataset)))
    print("Positive dataset size: " + str(len(positive_tweets)))
    print("Negative dataset size: " + str(len(negative_tweets)))

    return  selected_dataset


# relabel dataset by classification type
def relabel_by_classification_type(all_data, classification_type):
    if classification_type == 'multi':
        sentiment_mapper = {"sentiment": {"joy": 0, "sadness": 1, "fear": 2, "anger": 3,
                                          "love": 4, "surprise": 5
                                          }
                            }
        all_data.replace(sentiment_mapper, inplace=True)
        labels = ['joy', 'sadness', 'fear', 'anger', 'love', 'surprise']
        return all_data, labels
    elif classification_type == 'sentiment':
        sentiment_mapper = {"sentiment": {"joy": 1, "sadness": 0, "fear": 0, "anger": 0,
                                          "love": 1, "surprise": 1
                                          }
                            }
        all_data.replace(sentiment_mapper, inplace=True)
        labels = ['positive', 'negative']
        return all_data, labels
    elif classification_type == 'JoySadness':
        sentiment_mapper = {"sentiment": {"joy": 1, "sadness": 0
                                          }
                            }
        all_data.replace(sentiment_mapper, inplace=True)
        labels = ['Joy', 'Sadness']

        return all_data, labels
    elif classification_type == 'LoveSurprise':
        sentiment_mapper = {"sentiment": {"love": 1, "surprise": 0
                                          }
                            }
        all_data.replace(sentiment_mapper, inplace=True)
        labels = ['Love', 'Surprise']

        return all_data, labels
    elif classification_type == 'FearAnger':
        sentiment_mapper = {"sentiment": {"fear": 1, "anger": 0
                                          }
                            }
        all_data.replace(sentiment_mapper, inplace=True)
        labels = ['Fear', 'Anger']

        return all_data, labels
    pass


def remove_hashtags(sentence):
    new_sentence = ''
    # splitting the text into words
    for word in sentence.split():

        # checking the first character of every word
        if word[0] != '#':
            #if word.startswith('#'):
            #    print("Error")
            new_sentence = new_sentence + word+' '
        #else:
        #    print(word)
    return new_sentence
def sentence_preprocessor(sentence):
    sentence = sentence.lower()
    #sentence = convert_emoticons(sentence)
    sentence = emoji.demojize(sentence)
    sentence = remove_hashtags(sentence)

    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = remove_valid_links(sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub('[0-9]+', '', sentence)
    sentence = sentence.replace("https t", " ")
    sentence =  sentence.replace(" rt ", " ")
    encoded_string = sentence.encode("ascii", "ignore")
    sentence = encoded_string.decode()
    new_sentence = ""
    words = sentence.split()
    for word in words:
        #word = stem_word(word)
        if word=='https':
            continue
        if (check_valid_word(word)):
            new_sentence += word + " "
    new_sentence = remove_uninformative_words(new_sentence)
    new_sentence = new_sentence.strip()
    if 'https' in new_sentence:
        print('Word found.')
    #print(new_sentence)
    return new_sentence




# TF_IDF vectorization of the
def tf_idf_vectorize(data):
    # create a corpus or dictionary of all texts first
    corpus = data.loc[:, "text"]
    # create TF-IDF vectorizer using unigrams only
    unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
    # fit the vectorizer to our dataset's corpus
    unigram_tf_idf = unigram_vectorizer.fit_transform(corpus)
    # return unigram_tf_idf
    #print(unigram_tf_idf)
    return unigram_tf_idf
def get_text_from_data(tweets):
    text =  ' '.join(tweets['text'])
    text = sentence_preprocessor(text)
    return text


# clean the data set and collect metadata
def preprocess_data(all_data, classification_type):
    #relabeled_data, labels = #relabel_by_classification_type(all_data, classification_type)
    metadata = getMetaData(all_data)
    labels = ['provaccine','antivaccine']
    # perform preprocessing in each tweets
    all_data.applymap(lambda s: sentence_preprocessor(s) if type(s) == str else s)
    #print(all_data)
    #create_wordcloud(all_data)
    return all_data, metadata, labels


# extract features from the cleaned data and combine it with metadata
def extract_and_combine_features(preprocessed_data, metadata, classification_type):
    create_wordcloud(preprocessed_data)
    #get output label
    Y = preprocessed_data.loc[:, 'sentiment']
    # get TF_IDF vector
    tf_idf_vector = tf_idf_vectorize(preprocessed_data)
    # convert it from sparse to dense
    tf_idf_vector = tf_idf_vector.todense()
    # make a dataframe of tf-idf frame
    tf_idf_frame = panda.DataFrame(tf_idf_vector)
    # combine TF_IDF dataframe with metadata
    combined_data_frame = panda.concat([tf_idf_frame,metadata], axis=1)
    #print(combined_data_frame)
    return combined_data_frame, Y



def run_neural_network(X_train, y_train, X_test, y_test, classification_type, labels):
    # Neural network
    #print(y_train)
    y_train_categorical = np_utils.to_categorical(y_train, 2)
    y_test_categorical = np_utils.to_categorical(y_test, 2)
    model = Sequential()
    model.add(Dense(30,  activation='relu' , input_dim=len(X_train.columns)))
    #model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Model Training

    model.fit(X_train, y_train_categorical, epochs=15, verbose=1)
    #get prediction of test set
    output = numpy.argmax(model.predict(X_test), axis=-1)
    # Index of top values
    #print(output)
    for value in output:
        print(value)
    print_results_by_classification_type(y_test, output, labels, classification_type)
    ''' 
    if classification_type=='multi':
        plot_multiclass_confusion_matrix(y_test, output, labels)
        print_multi_class_results(y_test, output, labels)
    else:
        accuracy = accuracy_score(y_test, output)
        print("Accuracy: " + str(accuracy))
        print(confusion_matrix(y_test, output,labels=[1,-1]).ravel())
        tn, fp, fn, tp = confusion_matrix(y_test, output, labels=[1,-1]).ravel()
        print("Recall/Sensitivity: " + str(round(tp / (tp + fn), 2)))
        print("Specificity: " + str(round(tn / (tn + fp), 2)))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, output, pos_label=1)

        print("AUC: " + str(metrics.auc(fpr, tpr)))
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test, y_test_categorical)
    print('Accuracy: %.2f' % (accuracy * 100))
    '''
#return string label from output values
def get_original_labels(true_class, predicted_class, labels):
    y_true = []
    y_pred = []
    for value in true_class:
       if value==1:
           #provaccine
           y_true.append(labels[0])
       else:
           # antivaccine
           y_true.append(labels[1])
    for value in predicted_class:

        if value == 1:
            # provaccine
            y_pred.append(labels[0])
        else:
            # antivaccine
            y_pred.append(labels[1])
    return y_true, y_pred
#plot binary confusion matrix
def plot_binary_confusion_matrix(true_class, predicted_class, labels):
    #print(true_class)
    y_true, y_pred = get_original_labels(true_class, predicted_class, labels)
    multilabel_confusion_matrix = confusion_matrix(y_true, y_pred,
                                                   labels=labels)
    print(multilabel_confusion_matrix)

    fig, ax = plot_confusion_matrix(conf_mat=multilabel_confusion_matrix,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    )
    plt.xticks(numpy.arange(0, 2), labels)

    ax.set(yticks=[-0.5, 0, 0.5, 1],

           yticklabels=['', labels[0], '', labels[1]])

    plt.show()

#plot multiclass confusion matrix
def plot_multiclass_confusion_matrix(true_class, predicted_class, labels):
    y_true, y_pred = get_original_labels(true_class, predicted_class, labels)
    multilabel_confusion_matrix = confusion_matrix(y_true, y_pred,
                                                   labels=labels)
    print(multilabel_confusion_matrix)

    fig, ax = plot_confusion_matrix(conf_mat=multilabel_confusion_matrix,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    )
    plt.xticks(numpy.arange(0, 6), labels)

    ax.set(#yticks=[-0.5, 0, 0.5, 1.0,  1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
           yticklabels =[ '', labels[0],  labels[1],  labels[2],   labels[3],  labels[4],  labels[5]]
           )
    plt.show()

def print_multi_class_results(original_label, predicted_label, labels):
    multilabel_confusion_mat = multilabel_confusion_matrix(original_label, predicted_label)
    print(multilabel_confusion_mat)
    tn = multilabel_confusion_mat[:, 0, 0]
    tp = multilabel_confusion_mat[:, 1, 1]
    fn = multilabel_confusion_mat[:, 1, 0]
    fp = multilabel_confusion_mat[:, 0, 1]
    accuracy_scores = (tp + tn) / (tp + tn + fp + fn)
    sensitivity_scores = tp / (tp + fn)
    specificity_scores = tn / (tn + fp)
    weights_of_class =  [695.0/2000, 591.0/2000, 224.0/2000, 275.0/2000, 159.0/2000, 66.0/2000]
    print("All accuracy scores: "+ str(accuracy_scores))
    print("All sensitivity scores: "+ str(sensitivity_scores))
    print("All specificity scores: "+ str(specificity_scores))
    print("Weighted Accuracy: " + str(numpy.average(accuracy_scores, weights=weights_of_class)))
    print("Unweighted Accuracy: " + str(numpy.average(accuracy_scores)))
    print("Weighted Sensitivity: " + str(numpy.average(sensitivity_scores, weights=weights_of_class)))
    print("Unweighted Sensitivity: " + str(numpy.average(sensitivity_scores)))
    print("Weighted specificity: " + str(numpy.average(specificity_scores, weights=weights_of_class)))
    print("Unweighted specificity: " + str(numpy.average(specificity_scores)))

#print detailed results based on classification type
def print_results_by_classification_type(original_label, predicted_label, labels, classification_type):
    if classification_type == 'multi':
        plot_multiclass_confusion_matrix(original_label, predicted_label, labels)
        print_multi_class_results(original_label, predicted_label, labels)

    else:
        plot_binary_confusion_matrix(original_label, predicted_label, labels)
        accuracy = accuracy_score(original_label, predicted_label)
        print("Accuracy: " + str(accuracy))

        tn, fp, fn, tp = confusion_matrix(original_label, predicted_label, labels=[1, -1]).ravel()
        print( confusion_matrix(original_label, predicted_label).ravel())
        print("Recall/Sensitivity: " + str(round(tp / (tp + fn), 2)))
        print("Specificity: " + str(round(tn / (tn + fp), 2)))
        fpr, tpr, thresholds = metrics.roc_curve(original_label, predicted_label, pos_label=1)
        print("AUC: "+ str(metrics.auc(fpr, tpr)))


#run supervised ML methods sequentially on the combined data frame
def run_supervised_ML(combined_data, Y,  labels, classification_type):
    print(combined_data)
    # separate train_X, test_X, train_Y, test_Y by index
    train_X, test_X, train_Y, test_Y = train_test_split(
        combined_data, Y, test_size=0.25, random_state=42)
    #print(train_X)
    #print(test_X)
    #print(train_Y)
    #print(test_Y)
    #Run neural network by keras
    #run_neural_network(train_X, train_Y, test_X, test_Y, classification_type, labels)

    print("\nRF")
    clf = RandomForestClassifier(n_estimators=30)
    clf.fit(train_X, train_Y)
    print(clf.predict(test_X))
    y_pred = clf.predict(test_X)
    print_results_by_classification_type(test_Y, y_pred, labels, classification_type)


    print("\nMLP")
    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(10, 5, 2), max_iter=1000)
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    print_results_by_classification_type(test_Y, y_pred, labels, classification_type)

    print("\nGB")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0)
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    print_results_by_classification_type(test_Y, y_pred, labels, classification_type)

    print("\nSVM")
    clf = svm.SVC(kernel='linear')
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    accuracy_by_SVM = accuracy_score(test_Y, y_pred)
    print_results_by_classification_type(test_Y, y_pred, labels, classification_type)



#nltk.download()
#set classification type
classification_type = 'binary'
# get train, test and combined dataset
all_data = get_dataset(classification_type)
print(all_data)
#create_wordcloud(train_data)
# preprocess all data
preprocessed_data, metadata, labels = preprocess_data(all_data, classification_type)
#create_wordcloud(preprocessed_data)

# extract and combine features
combined_data, Y = extract_and_combine_features(preprocessed_data, metadata, classification_type)
print(preprocessed_data)
run_supervised_ML(combined_data, Y,  labels, classification_type)