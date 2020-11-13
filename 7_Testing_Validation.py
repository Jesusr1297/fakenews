# ## Custom News Tests

from datetime import datetime

import re
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
                                            strip_numeric, remove_stopwords, strip_short
import numpy as np
import pickle


def remove_url(s):
    regex = re.compile(r'https?://\S+|www\.\S+|bit\.ly\S+')
    return regex.sub(r'', s)


def return_vector(model_made, x):
    try:
        return model_made[x]
    except:
        return np.zeros(100)


def sentence_vector(model_made, sentence):
    word_vectors = list(map(lambda x: return_vector(model_made, x), sentence))
    return np.average(word_vectors, axis=0).tolist()


start = datetime.now()

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, remove_url, strip_punctuation, strip_multiple_whitespaces,
                  strip_numeric, remove_stopwords, strip_short]

processed_data = np.load('data/processed_data.npy', allow_pickle=True)
model = pickle.load(open('models/finalized_model.sav', 'rb'))
kmeans = pickle.load(open('models/kmeans.sav', 'rb'))

# Testing with fake news generated from https://www.thefakenewsgenerator.com/
onion_data = "Flint Residents Learn To Harness Superpowers, But Trump Gets Away Again They developed superpowers after years of drinking from a lead-poisoned water supply. But just having incredible abilities doesn't make them superheroes. Not yet. Donald Trump faced off against the superpowered civilians but he got away before they could catch him"

# Preprocess article
onion_data = preprocess_string(onion_data, CUSTOM_FILTERS)

# Get sentence vector
onion_data = sentence_vector(model, onion_data)

# Get prediction
print(kmeans.predict(np.array([onion_data])))


# News from BBC
bbc_data = "Nasa Mars 2020 Mission's MiMi Aung on women in space Next year, Nasa will send a mission to Mars. The woman in charge of making the helicopter that will be sent there – which is set to become the first aircraft to fly on another planet – is MiMi Aung. At 16, MiMi travelled alone from Myanmar to the US for access to education. She is now one of the lead engineers at Nasa. We find out what it's like being a woman in space exploration, and why her mum is her biggest inspiration."

# Preprocess article
bbc_data = preprocess_string(bbc_data, CUSTOM_FILTERS)

# Get sentence vector
bbc_data = sentence_vector(model, bbc_data)

# Get prediction
print(kmeans.predict(np.array([bbc_data])))

finish = datetime.now()
print(finish - start)
