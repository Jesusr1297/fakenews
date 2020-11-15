"""
   Once we have developed our visualization models
   it's time to plot them.

   The color palettes were obtained from https://colorhunt.co/palette/206723
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load data generated in 4_PCA and 5_TSNE
PCA_df = pd.read_csv('data/PCA.csv')
TSNE_df = pd.read_csv('data/TSNE.csv')


# setting style of plot
sns.set_style("darkgrid")
# defining the colors for our plot
background_color = '#eeecda'
lines_color = '#b83b5e'
dots_color = ['#6a2c70', '#f08a5d']

# setting plot color characteristics
sns.set('notebook', rc={'axes.facecolor': background_color, 'figure.facecolor': background_color,
                        'lines.color': lines_color, 'xtick.color': lines_color, 'ytick.color': lines_color,
                        'axes.labelcolor': lines_color, 'text.color': lines_color})

# configuring a two plots in one image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# plotting first scatter (PCA) displayed on the right side
sns.scatterplot(data=PCA_df, x='x1', y='x2', hue='cluster',
                legend="full", alpha=1, ax=ax[1], palette=dots_color,
                edgecolors=None)
# plotting second scatter (TSNE) displayed on the left side
sns.scatterplot(data=TSNE_df, x='x1', y='x2', hue='cluster',
                legend="full", alpha=1, ax=ax[0], palette=dots_color,
                edgecolors=None)
# adding titles
ax[0].set_title('Visualized on TSNE')
ax[1].set_title('Visualized on PCA')
# saving the plot
plt.savefig('plots/PCA_TSNE_5.jpg', bbox_inches='tight')
# showing results
plt.show()
