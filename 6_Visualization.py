# ## Visualization

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


PCA_df = pd.read_csv('data/PCA.csv')
TSNE_df = pd.read_csv('data/TSNE.csv')

color_list = [sns.color_palette('rocket', 2), ['darkred', 'indigo']]

# Plots

# https://colorhunt.co/palette/206723
sns.set_style("darkgrid")
background_color = '#eeecda'
lines_color = '#b83b5e'
dots_color = ['#6a2c70', '#f08a5d']

sns.set('notebook', rc={'axes.facecolor': background_color, 'figure.facecolor': background_color,
                        'lines.color': lines_color, 'xtick.color': lines_color, 'ytick.color': lines_color,
                        'axes.labelcolor': lines_color, 'text.color': lines_color})

fig, ax = plt.subplots(1, 2, figsize=(12,6))
sns.scatterplot(data=PCA_df, x='x1', y='x2', hue='cluster',
                legend="full", alpha=1, ax=ax[1], palette=dots_color,
                edgecolors=None)
sns.scatterplot(data=TSNE_df, x='x1', y='x2', hue='cluster',
                legend="full", alpha=1, ax=ax[0], palette=dots_color,
                edgecolors=None)

ax[0].set_title('Visualized on TSNE')
ax[1].set_title('Visualized on PCA')
plt.savefig('plots/PCA_TSNE_5.jpg', bbox_inches='tight')
plt.show()
