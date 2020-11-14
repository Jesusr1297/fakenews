import pickle

kmeans = pickle.load(open('models/kmeans.pkl', 'rb'))
news_list = pickle.load(open('news/news_list.pkl', 'rb'))

print(kmeans.predict(news_list))
