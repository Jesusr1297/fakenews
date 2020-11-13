# ## Word2Vec

from gensim.models import Word2Vec
import numpy as np
import pickle

# Time control
from datetime import datetime

start = datetime.now()

loaded = np.load('data/processed_data.npy', allow_pickle=True)
processed_data = loaded.tolist()

# Word2Vec model trained on processed data
model = Word2Vec(processed_data, min_count=1)


# ## Sentence Vectors
# Getting the vector of a sentence based on average of all the word vectors in the sentence
# We get the average as this accounts for different sentence lengths


def return_vector(model_made, x):
    try:
        return model_made[x]
    except:
        return np.zeros(100)


def sentence_vector(model_made, sentence):
    word_vectors = list(map(lambda x: return_vector(model_made, x), sentence))
    return np.average(word_vectors, axis=0).tolist()


X = [sentence_vector(model, data) for data in processed_data]
np.save('data/X', X)
pickle.dump(model, open('models/model.sav', 'wb'))

finish = datetime.now()
print(finish - start)
