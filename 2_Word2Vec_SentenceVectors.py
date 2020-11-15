"""
    In previous code we obtained a list of lists of words,
    in reality, we training a neural network, we want the strings
    as numbers, this code does this job with Word2Vec function
    we then pass the list of lists of numbers to a sentences vector
"""

from gensim.models import Word2Vec
import numpy as np
import pickle
from functions import sentence_vector

# load processed data generated in 1_DataAnalysis_Cleanup
loaded = np.load('data/processed_data.npy', allow_pickle=True)
# pass the np.array into a list
processed_data = loaded.tolist()

# Word2Vec model trained on processed data
model = Word2Vec(processed_data, min_count=1)


# ## Sentence Vectors
# Getting the vector of a sentence based on average of all the word vectors in the sentence
# We get the average as this accounts for different sentence lengths
X = [sentence_vector(model, data) for data in processed_data]

# save the list generated list
np.save('data/X', X)
pickle.dump(model, open('models/model.pkl', 'wb'))
