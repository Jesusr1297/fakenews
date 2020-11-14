import pickle
import re
import numpy as np
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
                                            strip_numeric, remove_stopwords, strip_short


model = pickle.load(open('models/model.pkl', 'rb'))


# 1_DataAnalysis_Cleanup
def remove_url(s):
    regex = re.compile(r'https?://\S+|www\.\S+|bit\.ly\S+')
    return regex.sub(r'', s)


CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, remove_url, strip_punctuation, strip_multiple_whitespaces,
                  strip_numeric, remove_stopwords, strip_short]


# 2_Word2Vec_SentenceVectors
def return_vector(model_made, x):
    try:
        return model_made[x]
    except:
        return np.zeros(100)


def sentence_vector(model_made, sentence):
    word_vectors = list(map(lambda x: return_vector(model_made, x), sentence))
    return np.average(word_vectors, axis=0).tolist()


# 7_TestNewsProcessing
def prepare_news(news=None, method=sentence_vector):
    if type(news) is not list:
        news = [news]
    return np.array([method(model, preprocess_string(new, CUSTOM_FILTERS)) for new in news])
