from datetime import datetime

import pandas as pd
import re
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
                                            strip_numeric, remove_stopwords, strip_short

import numpy as np
from gensim.models import Word2Vec

from sklearn import cluster

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

start = datetime.now()

fake = pd.read_csv('../data/Fake.csv')
true = pd.read_csv('../data/True.csv')

print('Fake news preview')
print(fake.head(15))
print('\n\n\n True news preview')
print(true.head(15))

"""
The first issue as seen above is that the True data contains:

1. A reuters disclaimer that the article is a tweet
> "The following statements were posted to the verified Twitter accounts of U.S. President Donald Trump, @realDonaldTrump and @POTUS.  The opinions expressed are his own. Reuters has not edited the statements or confirmed their accuracy.  @realDonaldTrump"


2. City Name and then publisher at the start
> WASHINGTON (Reuters)

so in the next block of code I remove this from the data
"""

# The following is a crude way to remove the @realDonaldTrump tweet disclaimer and State/Publisher at start of text
cleansed_data = []
for data in true.text:
    if "@realDonaldTrump : - " in data:
        cleansed_data.append(data.split("@realDonaldTrump : - ")[1])
    elif "(Reuters) -" in data:
        cleansed_data.append(data.split("(Reuters) - ")[1])
    else:
        cleansed_data.append(data)

true["text"] = cleansed_data
print(true.head(10))

"""
Some of the text still contains various characters/words such as:

1. Links
2. Timestamps
3. Brackets
4. Numbers

So we will be removing all such characters from the real and fake data using genlib preprocessing and a custom regex for the links in preparation for the Word2Vec

Before that however, the title and text will be merged in to one so that it can all be preprocessed together. I will also add a label for real and fake which will be used later to evaluate our clustering
"""

# Merging title and text
fake['Sentences'] = fake['title'] + ' ' + fake['text']
true['Sentences'] = true['title'] + ' ' + true['text']

# Adding fake and true label
fake['Label'] = 0
true['Label'] = 1

# We can merge both together since we now have labels
final_data = pd.concat([fake, true])

# Randomize the rows so its all mixed up
final_data = final_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop columns not needed
final_data = final_data.drop(['title', 'text', 'subject', 'date'], axis=1)
print(final_data.head(10))


# Here we preprocess the sentences
def remove_url(s):
    regex = re.compile(r'https?://\S+|www\.\S+|bit\.ly\S+')
    return regex.sub(r'', s)


# Preprocessing functions to remove lowercase, links, whitespace, tags, numbers, punctuation, strip words
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, remove_url, strip_punctuation, strip_multiple_whitespaces,
                  strip_numeric, remove_stopwords, strip_short]

# Here we store the processed sentences and their label
words_broken_up = [preprocess_string(sentence, CUSTOM_FILTERS) for sentence in final_data.Sentences]
processed_data = [word for word in words_broken_up if len(word) > 0]
processed_labels = [label for num, label in enumerate(final_data.Label) if len(words_broken_up[num]) > 0]

print(processed_labels[:20])
print(processed_data[:20])


# Word2Vec model trained on processed data
model = Word2Vec(sentences=processed_data, min_count=1)

# ## Sentence Vectors
# Getting the vector of a sentence based on average of all the word vectors in the sentence
# We get the average as this accounts for different sentence lengths


def return_vector(model_made=None, x=None):
    try:
        return model_made[x]
    except:
        return np.zeros(100)


def sentence_vector(model_made, sentence):
    word_vectors = list(map(lambda x: return_vector(model_made, x), sentence))
    return np.average(word_vectors, axis=0).tolist()


X = [sentence_vector(model, data) for data in processed_data]

# Training for 2 clusters (Fake and Real)
kmeans = cluster.KMeans(n_clusters=2, verbose=1)

# Fit predict will return labels
clustered = kmeans.fit_predict(X)

testing_df = {'Sentence': processed_data, 'Labels': processed_labels, 'Prediction': clustered}
testing_df = pd.DataFrame(data=testing_df)

# print(testing_df.head(10))
testing_df['is_correct'] = np.logical_not(np.logical_xor(testing_df['Labels'], testing_df['Prediction']))
print(np.sum(testing_df.is_correct)/np.sum(len(testing_df.is_correct))*100)


pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

PCA_df = pd.DataFrame(pca_result)
PCA_df.columns = ['x1', 'x2']
PCA_df['cluster'] = clustered
print(PCA_df.head(20))


tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(pca_result)

TSNE_df = pd.DataFrame(tsne_result)
TSNE_df.columns = ['x1', 'x2']
TSNE_df['cluster'] = clustered
print(TSNE_df.head(20))


color_list = [['darkred', 'indigo'], sns.color_palette('rocket', 2), sns.color_palette('mako', 2)]

# Plots
for palette in color_list:

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.set()
    sns.scatterplot(data=PCA_df, x='x1', y='x2', hue='cluster', legend="full", alpha=0.5, ax=ax[1], palette=palette)
    sns.scatterplot(data=TSNE_df, x='x1', y='x2', hue='cluster', legend="full", alpha=0.5, ax=ax[0], palette=palette)
    ax[1].set_title('Visualized on PCA')
    ax[0].set_title('Visualized on TSNE')
    plt.show()


# Testing with fake news generated from https://www.thefakenewsgenerator.com/
onion_data = "Flint Residents Learn To Harness Superpowers, But Trump Gets Away Again They developed superpowers after years of drinking from a lead-poisoned water supply. But just having incredible abilities doesn't make them superheroes. Not yet. Donald Trump faced off against the superpowered civilians but he got away before they could catch him"

# Preprocess article
onion_data = preprocess_string(onion_data, CUSTOM_FILTERS)

# Get sentence vector
onion_data = sentence_vector(model, onion_data)

# Get prediction
print(kmeans.predict(np.array([onion_data])))


# News from BBC
bbc_data = "Nasa Mars 2020 Mission's MiMi Aung on women in space Next year, Nasa will send a mission to Mars. The woman in charge of making the helicopter that will be sent there – which is set to become the first aircraft to fly on another planet – is MiMi Aung. At 16, MiMi travelled alone from Myanmar to the US for access to education. She is now one of the lead engineers at Nasa. We find out what it's like being a woman in space exploration, and why her mum is her biggest inspiration."

# Preprocess article
bbc_data = preprocess_string(bbc_data, CUSTOM_FILTERS)

# Get sentence vector
bbc_data = sentence_vector(model, bbc_data)

# Get prediction
print(kmeans.predict(np.array([bbc_data])))

finish = datetime.now()
print(finish - start)
