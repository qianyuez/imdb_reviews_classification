import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import string


MAX_WORDS = 10000
RANDOM_STATE = 1


def divide_words(sentences):
    corpus = []
    stop_words = set(stopwords.words('english'))
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word for word in words if word not in stop_words and word not in string.punctuation]
        corpus.append(' '.join(words))
    return corpus


def load_data(path):
    df = pd.read_csv(path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    reviews = df['review'].values
    labels = df['sentiment'].values
    reviews = divide_words(reviews)

    reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, labels, random_state=RANDOM_STATE)
    countVectorizer = CountVectorizer(max_features=MAX_WORDS, binary=True)
    onehot_train = countVectorizer.fit_transform(reviews_train)
    onehot_test = countVectorizer.transform(reviews_test)

    return onehot_train, onehot_test, y_train, y_test
