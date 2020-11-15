"""
    When we train a neural network it is useful to visualize
    the results, PCA stands for Principal Component Analysis
    and it is a tool commonly used for dimensionality reduction.

    The way it work is by projecting each data point onto only
    the first few principal components to obtain lower-dimensional
    data while preserving as much of the data's variation as possible.
"""

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from functions import load_model
import pickle

# we load data generated in 2_Word2Vec_SentenceVectors and 3_Clustering
X = np.load('data/X.npy')
clustered = load_model('models/clustered.pkl')

# instantiate PCA object into two main components
pca = PCA(n_components=2)
# we fit (train) and transform into PCA
pca_result = pca.fit_transform(X)

# DataFrame creation with information generated
PCA_df = pd.DataFrame(pca_result)
# columns renamed
PCA_df.columns = ['x1', 'x2']
# create cluster column
PCA_df['cluster'] = clustered

# saving our DataFrame
PCA_df.to_csv('data/PCA.csv')
# displaying a 20 line preview of the DataFrame
print(PCA_df.head(20))

# saving pca model
pickle.dump(pca_result, open('models/PCA.sav', 'wb'))
