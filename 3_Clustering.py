import pandas as pd
import numpy as np
from sklearn import cluster

import pickle


# load data and labels generated in 1_DataAnalysis_Cleanup
processed_data = np.load('data/processed_data.npy', allow_pickle=True)
processed_labels = np.load('data/processed_labels.npy')

# load X list generated in 2_Word2Vec_SentenceVectors
X = np.load('data/X.npy')

# Training for 2 clusters (Fake and Real)
kmeans = cluster.KMeans(verbose=1, n_clusters=2)

# Fit predict will return labels
clustered = kmeans.fit_predict(X)

# test dictionary to create a DataFrame
test = {'Sentence': processed_data, 'Labels': processed_labels, 'Prediction': clustered}
# creating a df with previous dictionary
test_df = pd.DataFrame(test)
# printing first 25 rows to see results
print(test_df.head(25))

# create an is_correct column to determine if predicted value coincide with real value
test_df['is_correct'] = np.logical_not(np.logical_xor(test_df['Labels'], test_df['Prediction']))
# print percentage of accuracy
print(np.sum(test_df.is_correct) / np.sum(len(test_df.is_correct)) * 100)

# save generated models
pickle.dump(kmeans, open('models/kmeans.pkl', 'wb'))
pickle.dump(clustered, open('models/clustered.pkl', 'wb'))
