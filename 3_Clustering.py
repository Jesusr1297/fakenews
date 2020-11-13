# ## Clustering
import pandas as pd
import numpy as np
from sklearn import cluster

import pickle

processed_data = np.load('data/processed_data.npy', allow_pickle=True)
processed_labels = np.load('data/processed_labels.npy')
X = np.load('data/X.npy')
# Training for 2 clusters (Fake and Real)
kmeans = cluster.KMeans(n_clusters=2, verbose=1)

# Fit predict will return labels
clustered = kmeans.fit_predict(X)

testing_df = {'Sentence': processed_data, 'Labels': processed_labels, 'Prediction': clustered}
testing_df = pd.DataFrame(data=testing_df)

# print(testing_df.head(10))
testing_df['is_correct'] = np.logical_not(np.logical_xor(testing_df['Labels'], testing_df['Prediction']))
print(np.sum(testing_df.is_correct)/np.sum(len(testing_df.is_correct))*100)

pickle.dump(kmeans, open('models/kmeans.sav', 'wb'))
pickle.dump(clustered, open('models/clustered.sav', 'wb'))
