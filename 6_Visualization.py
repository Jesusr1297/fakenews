# ## Visualization

from datetime import datetime

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

start = datetime.now()

PCA_df = pd.read_csv('data/PCA.csv')
TSNE_df = pd.read_csv('data/TSNE.csv')

color_list = [sns.color_palette('rocket', 2), ['darkred', 'indigo']]

# Plots
for palette in color_list:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(data=PCA_df, x='x1', y='x2', hue='cluster', legend="full", alpha=0.5, ax=ax[1], palette=palette)
    sns.scatterplot(data=TSNE_df, x='x1', y='x2', hue='cluster', legend="full", alpha=0.5, ax=ax[0], palette=palette)
    ax[0].set_title('Visualized on TSNE')
    ax[1].set_title('Visualized on PCA')
    plt.savefig('plots/PCA_TSNE_2.jpg', bbox_inches='tight')
    plt.show()

finish = datetime.now()
print(finish - start)
