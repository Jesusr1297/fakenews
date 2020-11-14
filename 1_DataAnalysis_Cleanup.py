# todo comment
import pandas as pd
import numpy as np
from functions import CUSTOM_FILTERS

# Preprocessing
from gensim.parsing.preprocessing import preprocess_string


fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

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


# Here we store the processed sentences and their label
words_broken_up = [preprocess_string(sentence, CUSTOM_FILTERS) for sentence in final_data.Sentences]
processed_data = [word for word in words_broken_up if len(word) > 0]
processed_labels = [label for num, label in enumerate(final_data.Label) if len(words_broken_up[num]) > 0]

print(processed_labels[:20])
print(processed_data[:20])

np.save('data/processed_labels', processed_labels)
np.save('data/processed_data', processed_data)
