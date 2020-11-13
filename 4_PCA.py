# PCA

from datetime import datetime

from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import pickle

start = datetime.now()

X = np.load('data/X.npy')
clustered = pickle.load(open('models/clustered.sav', 'rb'))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

PCA_df = pd.DataFrame(pca_result)
PCA_df.columns = ['x1', 'x2']
PCA_df['cluster'] = clustered


PCA_df.to_csv('data/PCA.csv')
print(PCA_df.head(20))

pickle.dump(pca_result, open('models/PCA.sav', 'wb'))

finish = datetime.now()
print(finish - start)
