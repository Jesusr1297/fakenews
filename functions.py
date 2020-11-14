import pickle
import re
import numpy as np
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
                                            strip_numeric, remove_stopwords, strip_short

# we load the model for future use
model = pickle.load(open('models/model.pkl', 'rb'))


# 1_DataAnalysis_Cleanup
def remove_url(s):
    """
    Removes url from given sentence
    :param s: sentence to remove urls
    :return: clean sentence with no url
    """
    regex = re.compile(r'https?://\S+|www\.\S+|bit\.ly\S+')
    return regex.sub(r'', s)


# list of functions for data cleaning
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, remove_url, strip_punctuation, strip_multiple_whitespaces,
                  strip_numeric, remove_stopwords, strip_short]


# 2_Word2Vec_SentenceVectors
def return_vector(model_made, x):
    """
    Function to extract a vector from given model
    :param model_made: model to extract info
    :param x: exact point to extract from model
    :return: Vector for given model
    """
    try:
        # if vector exist, returns vector
        return model_made[x]
    except:
        # if not, returns a list of zeros (100)
        return np.zeros(100)


def sentence_vector(model_made, sentence):
    """
    Converts the returned vector into a sentence vector (value vector)
    :param model_made: model created
    :param sentence: data to vector
    :return: list of averaged word vectors
    """
    word_vectors = list(map(lambda x: return_vector(model_made, x), sentence))
    return np.average(word_vectors, axis=0).tolist()


# 7_TestNewsProcessing
def prepare_news(news=None, method=sentence_vector):
    """
    Converts given news into a number vector to be evaluated
    :param news: news to be converted, can be a list of strings or a single string
    :param method: how is going to be converted, default: sentence_vector
    :return: an array of vectorized news
    """
    if type(news) is not list:
        news = [news]
    return np.array([method(model, preprocess_string(new, CUSTOM_FILTERS)) for new in news])
