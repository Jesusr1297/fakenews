"""
    Another useful tool is the T-distributed Stochastic Neighbor Embedding (TSNE)
    a nonlinear dimensionality reduction technique well-suited for embedding
    high-dimensional data for visualization in a low-dimensional space of two or
    three dimensions.
"""

from sklearn.manifold import TSNE
import pandas as pd
from functions import load_model

# load data generated in 3_Clustering and 4_PCA
pca_result = load_model('models/PCA.sav')
clustered = load_model('models/clustered.pkl')

# instantiate TSNE object with two main clusters
tsne = TSNE(n_components=2)
# we fit (train) and transform into TSNE
tsne_result = tsne.fit_transform(pca_result)

# DataFrame creation with information generated
TSNE_df = pd.DataFrame(tsne_result)
# columns renamed
TSNE_df.columns = ['x1', 'x2']
# create cluster column
TSNE_df['cluster'] = clustered

# saving our DataFrame
TSNE_df.to_csv('data/TSNE.csv')
# displaying a 20 line preview of the DataFrame
print(TSNE_df.head(20))
