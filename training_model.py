#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib

# === SETTINGS ===
ORIG_PATH = "/Users/ojaswijoshi/Desktop/DSCI 441(Stat & ML)/Mental-health-assessment/reddit_multi_temporal.csv"
EXT_PATH  = "/Users/ojaswijoshi/Downloads/depression_dataset_reddit_cleaned.csv"

# --- Temporal features ---
temp_feats = [
    'posts_last_3d','avg_sent_last_3d',
    'posts_last_7d','avg_sent_last_7d',
    'posts_last_14d','avg_sent_last_14d',
    'posts_last_30d','avg_sent_last_30d',
    'posts_last_365d','avg_sent_last_365d',
    'days_since_prev'
]

# --- Expanded self-harm lexicon (regex patterns) ---
self_harm_terms = [
    r"suicid", r"kill myself", r"end my life", r"hurt myself", r"die by suicide",
    r"take my own life", r"jump off", r"no reason to live", r"slit my wrist",
    r"overdose", r"cut myself", r"self[- ]harm", r"don't want to live"
]

def detect_self_harm(text):
    txt = text.lower()
    return int(any(re.search(term, txt) for term in self_harm_terms))

# === 1) LOAD & PREPARE ORIGINAL DATA ===
df_orig = pd.read_csv(ORIG_PATH, parse_dates=['created_dt'])
df_orig.dropna(subset=['content'], inplace=True)
df_orig['content'] = df_orig['content'].str.lower()
df_orig[temp_feats] = df_orig[temp_feats].fillna(0)

print("Processing original data...")
df_orig['sentiment']   = df_orig['content'].apply(lambda t: TextBlob(t).sentiment.polarity)
df_orig['is_positive'] = (df_orig['sentiment'] > 0.3).astype(int)
df_orig['self_harm']   = df_orig['content'].apply(detect_self_harm)
# Now label only posts with explicit self_harm = 1
df_orig['label']       = df_orig['self_harm']
df_orig['upvote_z']    = (df_orig['upvotes'] - df_orig['upvotes'].mean())   / df_orig['upvotes'].std()
df_orig['comment_z']   = (df_orig['comments'] - df_orig['comments'].mean()) / df_orig['comments'].std()

# === 2) LOAD & PREPARE EXTERNAL DATA ===
df_ext = pd.read_csv(EXT_PATH)
df_ext.rename(columns={'clean_text':'content','is_depression':'label'}, inplace=True)
df_ext['content'] = df_ext['content'].str.lower()
# fill missing temporal for external with zeros
for c in temp_feats:
    df_ext[c] = 0.0
df_ext['sentiment']   = df_ext['content'].apply(lambda t: TextBlob(t).sentiment.polarity)
df_ext['is_positive'] = (df_ext['sentiment'] > 0.3).astype(int)
df_ext['self_harm']   = df_ext['content'].apply(detect_self_harm)
df_ext['upvote_z'], df_ext['comment_z'] = 0.0, 0.0

# === 3) SPLIT ORIGINAL & EXTERNAL ===
cut = int(0.8 * len(df_orig))
orig_train = df_orig.iloc[:cut].reset_index(drop=True)
orig_hold  = df_orig.iloc[cut:].reset_index(drop=True)
train_ext, hold_ext = train_test_split(df_ext, test_size=0.2,
                                       stratify=df_ext['label'], random_state=42)

# === 4) TRAIN TEXT PIPELINE ===
corpus = pd.concat([orig_train['content'], train_ext['content']], ignore_index=True)
vectorizer = CountVectorizer(stop_words='english', min_df=5)
dtm = vectorizer.fit_transform(corpus)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
topics = lda.fit_transform(dtm)

# assign topics back to train splits
topics_orig = topics[:len(orig_train)]
topics_ext  = topics[len(orig_train):]
for i in range(5):
    orig_train[f'topic_{i}'] = topics_orig[:, i]
    train_ext[f'topic_{i}']   = topics_ext[:, i]

# helper to add topics to holdouts
def add_topics(df):
    dtm_tmp = vectorizer.transform(df['content'])
    tops    = lda.transform(dtm_tmp)
    for j in range(5):
        df[f'topic_{j}'] = tops[:, j]
    return df

orig_hold = add_topics(orig_hold)
hold_ext  = add_topics(hold_ext)

# === 5) DEFINE FEATURES & MATRICES ===
FEATURES = [
    'sentiment','is_positive','self_harm','upvote_z','comment_z'
] + temp_feats + [f'topic_{i}' for i in range(5)]

train_comb = pd.concat([orig_train, train_ext], ignore_index=True)
X_train    = train_comb[FEATURES]
y_train    = train_comb['label']
X_hold_ext = hold_ext[FEATURES]
y_hold_ext = hold_ext['label']

# === 6) Logistic Regression Baseline ===
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
baseline = cross_val_score(lr, X_train, y_train, cv=5, scoring='f1').mean()
print(f"\nLogisticRegression CV F1: {baseline:.3f}")
lr.fit(X_train, y_train)
print("\nLR on external hold-out:")
print(classification_report(y_hold_ext, lr.predict(X_hold_ext), digits=3))

# === 7) HistGradientBoosting with GridSearch ===
hgb = HistGradientBoostingClassifier(
    max_iter=500, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, l2_regularization=1.0, random_state=42
)
param_grid = {
    'max_depth':      [3, 5],
    'learning_rate': [0.05, 0.1],
    'max_leaf_nodes':[31, 63]
}
grid = GridSearchCV(hgb, param_grid, scoring='f1', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

best_hgb = grid.best_estimator_
print(f"\nBest HGB params: {grid.best_params_}")
print("\nHGB on external hold-out:")
print(classification_report(y_hold_ext, best_hgb.predict(X_hold_ext), digits=3))

# === 8) Save Artifacts ===
joblib.dump(vectorizer, 'vectorizer_final.joblib')
joblib.dump(lda,        'lda_final.joblib')
joblib.dump(best_hgb,   'hgb_final.joblib')


