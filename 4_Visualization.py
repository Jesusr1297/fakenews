# ## Visualization
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import pickle

start = datetime.now()
X = np.load('data/X.npy')
kmeans = pickle.load(open('models/finalized_model.sav', 'rb'))
clustered = pickle.load(open('models/clustered.sav', 'rb'))


# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

PCA_df = pd.DataFrame(pca_result)
PCA_df.columns = ['x1', 'x2']
PCA_df['cluster'] = clustered

# T-SNE
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(pca_result)

TSNE_df = pd.DataFrame(tsne_result)
TSNE_df.columns = ['x1', 'x2']
TSNE_df['cluster'] = clustered


# Plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(data=PCA_df, x='x1', y='x2', hue='cluster', legend="full", alpha=0.5, ax=ax[1], palette=sns.color_palette('rocket', 2))      # sns.color_palette('rocket', 2)
sns.scatterplot(data=TSNE_df, x='x1', y='x2', hue='cluster', legend="full", alpha=0.5, ax=ax[0], palette=sns.color_palette('rocket', 2))     # ['darkred', 'indigo']
ax[0].set_title('Visualized on TSNE')
ax[1].set_title('Visualized on PCA')
plt.savefig('plots/PCA_TSNE_1.jpg', bbox_inches='tight')
plt.show()

finish = datetime.now()
print(finish - start)
