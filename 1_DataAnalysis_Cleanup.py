"""
    This code reads the data given and cleans it
    from deleting unwanted strings, deleting unwanted columns
    and continuing with passing the information obtained as a list of lists
"""

import pandas as pd
import numpy as np
from functions import CUSTOM_FILTERS
from gensim.parsing.preprocessing import preprocess_string

# reading information obtained
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

# Making first explorations
print('Fake news preview')
print(fake.head(15))
print('True news preview')
print(true.head(15))

"""
When we observe the true df, we can notice the following:

        1. A reuters disclaimer that the article is a tweet
        > "The following statements were posted to the verified Twitter accounts of U.S. President Donald Trump, @realDonaldTrump and @POTUS.  The opinions expressed are his own. Reuters has not edited the statements or confirmed their accuracy.  @realDonaldTrump"

        2. City Name and then publisher at the start
        > WASHINGTON (Reuters)

In the next lines we clean the unwanted information
"""

# The following is a crude but easy way to remove the @realDonaldTrump tweet disclaimer and State/Publisher at start of text
cleansed = []
for data in true['text']:
    """ We loop over every row in the text column and drop the unuseful string by splitting the sentence
    
    for example if, when we find that the sentence starts with or contains "@realDonaldTrump : -"
    we do the following
        We first split the data 
        > "@realDonaldTrump : - the US president told Barack Obama...".split("@realDonaldTrump : - "
        Once divided we select the second element of the list
        > ['@realDonaldTrump : -', 'the US president told Barack Obama...'][1]
        we append cleaned sentence in a new list
        > 'the US president told Barack Obama...'.append()
    """
    if "@realDonaldTrump : - " in data:
        cleansed.append(data.split("@realDonaldTrump : - ")[1])
    elif "(Reuters) -" in data:
        cleansed.append(data.split("(Reuters) - ")[1])
    else:
        cleansed.append(data)
# after cleaned the sentences, we add a column to our true DataFrame
# with the results
true["text"] = cleansed
# We inspect job was done correctly
print('True cleaned')
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

# Merging title and text into one line new
fake['Sentences'] = fake['title'] + ' ' + fake['text']
true['Sentences'] = true['title'] + ' ' + true['text']

# Adding fake and true label
fake['Label'] = 0
true['Label'] = 1

# We can merge both together since we now have labels
concatenated = pd.concat([fake, true])

# Randomize the rows so its all mixed up and resetting the index
concatenated = concatenated.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop columns not needed
concatenated = concatenated.drop(['title', 'text', 'subject', 'date'], axis=1)
# Inspect results
print('Concatenated')
print(concatenated.head(10))


# Here we store the processed sentences and their label in two different lists
# first we apply all the filters to every sentence and append to a list
words_broken_up = [preprocess_string(sentence, CUSTOM_FILTERS) for sentence in concatenated.Sentences]

# then, we pass every word in every sentence to a list, obtaining a list of lists
# this only if the word exists (len(word) > 0)
processed_data = [word for word in words_broken_up if len(word) > 0]
# we do the same for the labels
processed_labels = [label for num, label in enumerate(concatenated.Label) if len(words_broken_up[num]) > 0]

# printing to see results
print(processed_labels[:20])
print(processed_data[:20])

# saving results
np.save('data/processed_labels', processed_labels)
np.save('data/processed_data', processed_data)
