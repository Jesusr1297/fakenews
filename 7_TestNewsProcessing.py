"""
    It's time to see the accuracy of our model by
    evaluating our own news, this news were manually
    downloaded from internet and saved in a txt file.

    This script only pre process the news, same as did before
"""

import numpy as np
import pickle
from glob import glob
from functions import prepare_news, load_model

# load data generated in 1_DataAnalysis_Cleanup, 2_Word2Vec_SentenceVectors and 3_Clustering
processed_data = np.load('data/processed_data.npy', allow_pickle=True)
model = load_model('models/model.pkl')
kmeans = load_model('models/kmeans.pkl')

# getting a list of all news paths to analyze
glob_list = glob('news/*.txt')
# getting a list of all news in string format
news_list = [open(new, 'r', encoding='utf8').read() for new in glob_list]

# processing news and saving result
pickle.dump(prepare_news(news_list), open('news/news_list.pkl', 'wb'))
