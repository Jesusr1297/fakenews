"""
    Here we predict the now processed news
    and see results in a DataFrame
"""

import pandas as pd
from functions import load_model

# load data generated in 3_Clustering and 7_TestNewsProcessing
kmeans = load_model('models/kmeans.pkl')
news_list = load_model('news/news_list.pkl')

# printing predictions
prediction = kmeans.predict(news_list)
print(prediction)

# printing a DataFrame with results obtained
df = pd.DataFrame({'Sentence': ['news' + str(num + 1).zfill(2) for num, item in enumerate(news_list)], 'Prediction': prediction})
print(df)
