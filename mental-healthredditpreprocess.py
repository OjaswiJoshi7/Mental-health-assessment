import praw
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

reddit = praw.Reddit(
    client_id="jnArLq5IvQzEr70pExGQwQ",
    client_secret="yucUUU4SoASSWOOMJ1Jh_PcuiTpivg",
    user_agent="Mentalhealthpredictor"
)

#Fetch top posts from r/mentalhealth
subreddit = reddit.subreddit("mentalhealth")
for post in subreddit.hot(limit=5):
    print(post.title)



nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Fetch Reddit data and preprocess it
posts = []
for post in subreddit.hot(limit=100):  # Fetch 100 posts
    if post.selftext:  # Ensure the post has text
        cleaned_text = clean_text(post.selftext)
        posts.append([post.title, cleaned_text, post.score])

# Convert to a Pandas DataFrame
df = pd.DataFrame(posts, columns=['title', 'content', 'upvotes'])
print(df.head())

# Convert text into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['content'])

# Save processed data for model training
df.to_csv('reddit_data.csv', index=False)
