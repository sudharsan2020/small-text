from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from active_learning.data import SklearnDataSet

from examples.data.corpus_twenty_news import get_twenty_newsgroups_corpus


def get_train_test():

    train, test = get_twenty_newsgroups_corpus()
    return train, test


def preprocess_data(train, test):
    vectorizer = TfidfVectorizer(stop_words='english')

    x_train = normalize(vectorizer.fit_transform(train.data))
    x_test = normalize(vectorizer.transform(test.data))

    return SklearnDataSet(x_train, train.target), SklearnDataSet(x_test, test.target)
