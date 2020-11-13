# T-SNE

from datetime import datetime

from sklearn.manifold import TSNE

import pandas as pd
import pickle

start = datetime.now()

pca_result = pickle.load(open('models/PCA.sav', 'rb'))
clustered = pickle.load(open('models/clustered.sav', 'rb'))

tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(pca_result)

TSNE_df = pd.DataFrame(tsne_result)
TSNE_df.columns = ['x1', 'x2']
TSNE_df['cluster'] = clustered

TSNE_df.to_csv('data/TSNE.csv')
print(TSNE_df.head(20))

finish = datetime.now()
print(finish - start)
