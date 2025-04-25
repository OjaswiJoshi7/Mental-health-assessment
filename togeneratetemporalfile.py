#!/usr/bin/env python3
"""
1) Scrape multiple mental‐health subreddits (with author & timestamp)
2) Compute rolling temporal features (3,7,14,30d) + days_since_prev
3) Save an enriched CSV for modeling
"""

import praw
import pandas as pd
from textblob import TextBlob


reddit = praw.Reddit(
    client_id='jnArLq5IvQzEr70pExGQwQ',
    client_secret='yucUUU4SoASSWOOMJ1Jh_PcuiTpivg',
    user_agent='ai_mh_assessment:v1.0 (by u/bluesandbloops)'
)

#Scrape the subreddits
subs = ['mentalhealth','depression','suicidewatch','anxiety']
posts = []
for sub in subs:
    for post in reddit.subreddit(sub).new(limit=2000):
        text = post.selftext.strip() or post.title
        posts.append({
            'subreddit':  sub,
            'id':          post.id,
            'author':     str(post.author),
            'created_dt': pd.to_datetime(post.created_utc, unit='s', utc=True),
            'content':    text.lower(),
            'upvotes':    post.score,
            'comments':   post.num_comments
        })

df = pd.DataFrame(posts)

#Base sentiment & sort
df['sentiment'] = df['content'].apply(lambda t: TextBlob(t).sentiment.polarity)
df = df.sort_values(['author','created_dt']).reset_index(drop=True)

# Rolling‐window feature function (using .values to avoid reindex errors)
def add_time_window_features(df, w):
    cnt_col = f'posts_last_{w}d'
    avg_col = f'avg_sent_last_{w}d'
    grp = df.groupby('author')
    # count of prior posts
    temp_cnt = grp.rolling(f'{w}d', on='created_dt')['content'] \
                  .count().shift(1)
    df[cnt_col] = temp_cnt.values
    # avg sentiment of prior posts
    temp_avg = grp.rolling(f'{w}d', on='created_dt')['sentiment'] \
                  .mean().shift(1)
    df[avg_col] = temp_avg.values

# Add features for 3,7,14,30,365 days
for window in [3, 7, 14, 30,365]:
    add_time_window_features(df, window)

#Days since previous post
df['days_since_prev'] = (
    df.groupby('author')['created_dt']
      .diff().dt.total_seconds() / 86400.0
).fillna(0)

#Save enriched CSV
out = 'reddit_multi_temporal.csv'
df.to_csv(out, index=False)
print(f"Enriched data written to {out}")
