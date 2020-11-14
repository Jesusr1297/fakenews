# ## Custom News Tests

import numpy as np
import pickle
from glob import glob
from functions import prepare_news


processed_data = np.load('data/processed_data.npy', allow_pickle=True)
model = pickle.load(open('models/model.sav', 'rb'))
kmeans = pickle.load(open('models/kmeans.pkl', 'rb'))

glob_list = glob('news/*.txt')
news_list = [open(new, 'r', encoding='utf8').read() for new in glob_list]

pickle.dump(prepare_news(news_list), open('news/news_list.pkl', 'wb'))
