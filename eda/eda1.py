# Loading modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_category(x):
    if 5 < x < 20:
        return 'low'
    elif 20 <= x < 40:
        return 'medium-low'
    elif 40 <= x < 60:
        return 'medium'
    elif 60 <= x < 80:
        return 'medium-high'
    elif 80 <= x:
        return 'high'
    else:
        return 'None'


# List of csv locations
csv_paths = ['../datasets/gossip_real.csv', '../datasets/gossip_fake.csv', '../datasets/politi_real.csv', '../datasets/politi_fake.csv']
# Name list of dataframes
name_list = ['gossip_real', 'gossip_fake', 'political_real', 'political_fake']
# Reading all 4 dataframes and appending in a list (list of dataframes)
df_list = [pd.read_csv(csv, usecols=['news_url']) for csv in csv_paths]

for num, df in enumerate(df_list):
    # Cleaning/erasing first part of url list in every frame
    df['news_url'] = df['news_url'].str.replace('https', 'http')
    df['news_url'] = df['news_url'].str.replace('http://', '')
    df['news_url'] = df['news_url'].str.replace('www.', '')
    df['webpage'] = df['news_url'].str.split('/').str[0]

    # Printing information of the data (data types, info distribution, memory, number of unique webpages)
    print(name_list[num])
    print(df.info())
    print(df.describe())
    print(df.webpage.value_counts())

    # plotting top 15 pages with most news published (fake or real)
    plot = sns.barplot(x=df.webpage.value_counts().head(15).index, y=df.webpage.value_counts().head(15), palette=sns.color_palette("magma"))
    plt.xticks(rotation=90)
    plt.ylabel(name_list[num])
    plt.savefig('top15_' + name_list[num] + '.jpg', bbox_inches='tight')
    plt.show()

# Dropping unuseful data from our data frames
df_list = [df.drop(columns='news_url') for df in df_list]
# Concatenating all dataframes
df_concat = pd.concat(df_list)
# Counting the number of total news published by webpage
web_list = [df.webpage.value_counts() for df in df_list]

# Concatenating news per webpage per kind into columns
frequency_df = pd.concat(web_list, axis=1)
# Changing the names of the columns
frequency_df.columns = name_list
# Changing NaN number (not a number) cells to 0 for math purposes
frequency_df[name_list] = frequency_df[name_list].fillna(0)

# Creating a column with the percentage of fake news published
frequency_df['fakenews_ratio'] = 100*(frequency_df['gossip_fake'] + frequency_df['political_fake']) / np.sum(frequency_df, axis=1)
# Creating a column with the category (high-medium-low...) depending on percentage of fakenews published
frequency_df['fakenews_cat'] = frequency_df['fakenews_ratio'].apply(get_category)

# Bar plotting top 25 webpages with the highest rate of fakenews published
bar = sns.barplot(x=frequency_df.fakenews_ratio.nlargest(25).index, y=frequency_df.fakenews_ratio.nlargest(25), palette=sns.color_palette("magma"))
plt.xticks(rotation=90)
plt.title('Sites with 100% of fakenews')
plt.savefig('bar.jpg', bbox_inches='tight')
plt.show()

# Counting the number of fakenews per category
count = sns.countplot(x='fakenews_cat', data=frequency_df, order=['None', 'low', 'medium-low', 'medium', 'medium-high', 'high'], palette=sns.color_palette("magma"))
plt.title('Number of sites vs the rate of fakenews published')
plt.savefig('count.jpg', bbox_inches='tight')
plt.show()

# Distribution of fakenews per category
dist = sns.displot(x=np.sum(frequency_df, axis=1), color='r', stat='density')
plt.xlim((0, 250))
plt.title('Distribution of fakenews published')
plt.savefig('dist.jpg', bbox_inches='tight')
plt.show()

# Scatter plot of fakenews  vs total news published
scatter = sns.scatterplot(x=np.sum(frequency_df, axis=1), y=frequency_df['political_fake'] + frequency_df['gossip_fake'])
plt.ylabel('Number of fake news published')
plt.xlabel('Number of news published')
plt.savefig('scatter.jpg', bbox_inches='tight')
plt.show()
