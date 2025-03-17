import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk
import nltk
from wordcloud import WordCloud
from collections import Counter
from scipy.stats import shapiro, kstest, mannwhitneyu, kruskal, chi2_contingency
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
if '/Users/ojaswijoshi/Desktop/DSCI 441(Stat & ML)/Mental-health-assessment/nltk_data' in nltk.data.path:
    nltk.data.path.remove('/Users/ojaswijoshi/Desktop/DSCI 441(Stat & ML)/Mental-health-assessment/nltk_data')
# Load the new dataset with 1000 Reddit posts
df = pd.read_csv('/Users/ojaswijoshi/Desktop/DSCI 441(Stat & ML)/Mental-health-assessment/reddit_data1000.csv')

# Handle missing values
df.dropna(subset=['content'], inplace=True)

# Compute word, char, and sentence count
df['word_count'] = df['content'].apply(
    lambda x: len(word_tokenize(str(x), language='english'))
)
df['char_count'] = df['content'].apply(lambda x: len(str(x)))
df['sentence_count'] = df['content'].apply(lambda x: str(x).count('.'))

# Compute sentiment polarity for each post
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity  # Returns sentiment score (-1 to 1)

df['sentiment'] = df['content'].apply(get_sentiment)

#Binning Sentiment into Categories
df['sentiment_category'] = pd.cut(df['sentiment'], bins=[-1, -0.3, 0.3, 1], labels=['Negative', 'Neutral', 'Positive'])

# Binning Upvotes into engagement levels: Low, Medium, High
df['engagement_level'] = pd.qcut(df['upvotes'], q=3, labels=['Low', 'Medium', 'High'])

# Common Words WordCloud
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

words = [word.lower() for post in df['content'] for word in word_tokenize(post) if word.isalpha() and word not in stop_words]
word_counts = Counter(words)

# Generate WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most Common Words in Mental Health Posts')
plt.show()

# Sentiment vs. Engagement Bar Chart
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='sentiment_category', hue='engagement_level', palette='coolwarm')
plt.title('Engagement Level by Sentiment Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.legend(title='Engagement Level')
plt.show()

# Upvotes by Sentiment Category
plt.figure(figsize=(8,5))
sns.boxplot(x=df['sentiment_category'], y=df['upvotes'])
plt.title('Upvotes vs Sentiment Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Upvotes')
plt.show()

# Word Count Distribution by Sentiment Category
plt.figure(figsize=(8,5))
sns.boxplot(x=df['sentiment_category'], y=df['word_count'])
plt.title('Word Count vs Sentiment Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Word Count')
plt.show()

# Feature Correlation Matrix
plt.figure(figsize=(8,5))
sns.heatmap(df[['upvotes', 'word_count', 'char_count', 'sentence_count', 'sentiment']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Histogram of Upvotes Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['upvotes'], bins=30, kde=True)
plt.title('Upvote Distribution')
plt.xlabel('Upvotes')
plt.ylabel('Frequency')
plt.show()

# Mann-Whitney U Test: Compare upvotes for negative vs. positive sentiment categories
low_sentiment = df[df['sentiment_category'] == 'Negative']['upvotes']
high_sentiment = df[df['sentiment_category'] == 'Positive']['upvotes']

mann_whitney = mannwhitneyu(low_sentiment, high_sentiment, alternative='two-sided')
print(f"Mann-Whitney U Test: {mann_whitney}")

# Kruskal-Wallis Test: Compare upvotes across all sentiment categories
kruskal_test = kruskal(
    df[df['sentiment_category'] == 'Negative']['upvotes'],
    df[df['sentiment_category'] == 'Neutral']['upvotes'],
    df[df['sentiment_category'] == 'Positive']['upvotes']
)
print(f"Kruskal-Wallis Test: {kruskal_test}")

# Chi-Square Test: Sentiment vs Engagement
contingency_table = pd.crosstab(df['sentiment_category'], df['engagement_level'])
chi2_test = chi2_contingency(contingency_table)
print(f"Chi-Square Test for Sentiment vs Engagement: {chi2_test}")

#1. Baseline demonstration

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Labeling 'at_risk' based on sentiment score
def label_at_risk(sentiment, threshold=-0.3):
    return 1 if sentiment < threshold else 0

df['at_risk'] = df['sentiment'].apply(label_at_risk)

print("=== Distribution of 'at_risk' label ===")
print(df['at_risk'].value_counts(normalize=True))
print()


majority_class = df['at_risk'].value_counts().idxmax()
df['baseline_majority_prediction'] = majority_class

acc_maj = accuracy_score(df['at_risk'], df['baseline_majority_prediction'])
prec_maj = precision_score(df['at_risk'], df['baseline_majority_prediction'], zero_division=0)
rec_maj = recall_score(df['at_risk'], df['baseline_majority_prediction'], zero_division=0)
f1_maj = f1_score(df['at_risk'], df['baseline_majority_prediction'], zero_division=0)

print("=== Baseline 1: Majority Class ===")
print(f"Majority Class: {majority_class}")
print(f"Accuracy:  {acc_maj:.2f}")
print(f"Precision: {prec_maj:.2f}")
print(f"Recall:    {rec_maj:.2f}")
print(f"F1 Score:  {f1_maj:.2f}\n")

# Rule-based classifier

def rule_based_classifier(sentiment, threshold=-0.5):
    return 1 if sentiment < threshold else 0

df['baseline_rule_prediction'] = df['sentiment'].apply(rule_based_classifier)

acc_rule = accuracy_score(df['at_risk'], df['baseline_rule_prediction'])
prec_rule = precision_score(df['at_risk'], df['baseline_rule_prediction'], zero_division=0)
rec_rule = recall_score(df['at_risk'], df['baseline_rule_prediction'], zero_division=0)
f1_rule = f1_score(df['at_risk'], df['baseline_rule_prediction'], zero_division=0)

print("=== Baseline 2: Rule-Based on Sentiment (threshold = -0.5) ===")
print(f"Accuracy:  {acc_rule:.2f}")
print(f"Precision: {prec_rule:.2f}")
print(f"Recall:    {rec_rule:.2f}")
print(f"F1 Score:  {f1_rule:.2f}\n")
