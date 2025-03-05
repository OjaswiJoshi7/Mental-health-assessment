import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk
nltk.download('punkt') 
nltk.download('punkt_tab')
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords

nltk.download('punkt')

# Load dataset
df = pd.read_csv('reddit_data.csv')

# Show basic info
print(df.info())

# Display first few rows
print(df.head())

#compute word and sentences count
df['word_count'] = df['content'].apply(lambda x: len(word_tokenize(x)))
df['char_count'] = df['content'].apply(lambda x: len(x))
df['sentence_count'] = df['content'].apply(lambda x: x.count('.'))

print(df[['word_count', 'char_count', 'sentence_count']].describe())

#plot of distribution of upvotes
plt.figure(figsize=(8,5))
sns.histplot(df['upvotes'], bins=30, kde=True)
plt.title('Distribution of Upvotes')
plt.xlabel('Upvotes')
plt.ylabel('Count')
plt.show()

#to check if longer posts get more upvotes
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['word_count'], y=df['upvotes'])
plt.title('Post Length vs. Upvotes')
plt.xlabel('Word Count')
plt.ylabel('Upvotes')
plt.show()

# Calculate correlation
correlation = df[['word_count', 'upvotes']].corr()
print("Correlation between word count and upvotes:\n", correlation)

#compute sentiment polarity for each post
from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity  # Returns sentiment score (-1 to 1)

df['sentiment'] = df['content'].apply(get_sentiment)

plt.figure(figsize=(8,5))
sns.histplot(df['sentiment'], bins=30, kde=True)
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.show()

print(df[['sentiment', 'upvotes']].corr())  # Check correlation

#frequent words in mental health post 


text = " ".join(df['content'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most Common Words in Mental Health Posts')
plt.show()

#print top 10 common words
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
words = [word.lower() for post in df['content'] for word in word_tokenize(post) if word.isalpha() and word not in stop_words]
word_counts = Counter(words)

# Print top 10 words
print(word_counts.most_common(10))

#correlation heatmap 
plt.figure(figsize=(8,5))
sns.heatmap(df[['upvotes', 'word_count', 'char_count', 'sentence_count', 'sentiment']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

#do positive or negative posts get more engagement?
plt.figure(figsize=(8,5))
sns.boxplot(x=pd.cut(df['sentiment'], bins=[-1, -0.5, 0, 0.5, 1]), y=df['upvotes'])
plt.title('Upvotes vs Sentiment')
plt.xlabel('Sentiment Category')
plt.ylabel('Upvotes')
plt.show()
