#!/usr/bin/env python3
"""
Train a combined pipeline using SentenceTransformer embeddings + self-harm,
sentiment, engagement and temporal features, then save the final model.
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib

# === SETTINGS ===
ORIG_PATH = "/Users/ojaswijoshi/Desktop/DSCI 441(Stat & ML)/Mental-health-assessment/reddit_multi_temporal.csv"
EXT_PATH  = "/Users/ojaswijoshi/Downloads/depression_dataset_reddit_cleaned.csv"

# --- Temporal features (must match your CSV) ---
temp_feats = [
    'posts_last_3d','avg_sent_last_3d',
    'posts_last_7d','avg_sent_last_7d',
    'posts_last_14d','avg_sent_last_14d',
    'posts_last_30d','avg_sent_last_30d',
    'posts_last_365d','avg_sent_last_365d',
    'days_since_prev'
]

# --- Self-harm lexicon (regex patterns) ---
self_harm_terms = [
    r"suicid", r"kill myself", r"end my life", r"hurt myself", r"die by suicide",
    r"take my own life", r"jump off", r"no reason to live", r"slit my wrist",
    r"overdose", r"cut myself", r"self[- ]harm", r"don't want to live"
]
def detect_self_harm(text: str) -> int:
    t = text.lower()
    return int(any(re.search(term, t) for term in self_harm_terms))

# === 1) LOAD & PREP ORIGINAL DATA ===
df_orig = pd.read_csv(ORIG_PATH, parse_dates=['created_dt'])
df_orig.dropna(subset=['content'], inplace=True)
df_orig['content'] = df_orig['content'].str.lower()
df_orig[temp_feats] = df_orig[temp_feats].fillna(0)

# numeric features on orig
df_orig['sentiment']   = df_orig['content'].apply(lambda t: TextBlob(t).sentiment.polarity)
df_orig['is_positive'] = (df_orig['sentiment'] > 0.3).astype(int)
df_orig['self_harm']   = df_orig['content'].apply(detect_self_harm)
# label = explicit self-harm only
df_orig['label']       = df_orig['self_harm']
df_orig['upvote_z']    = (df_orig['upvotes'] - df_orig['upvotes'].mean())   / df_orig['upvotes'].std()
df_orig['comment_z']   = (df_orig['comments'] - df_orig['comments'].mean()) / df_orig['comments'].std()

# split orig → 80/20 temporal
cut = int(0.8 * len(df_orig))
orig_train = df_orig.iloc[:cut].reset_index(drop=True)
orig_hold  = df_orig.iloc[cut:].reset_index(drop=True)

# === 2) LOAD & PREP EXTERNAL DATA ===
df_ext = pd.read_csv(EXT_PATH)
df_ext.rename(columns={'clean_text':'content','is_depression':'label'}, inplace=True)
df_ext['content'] = df_ext['content'].str.lower()
df_ext[temp_feats] = 0.0
df_ext['sentiment']   = df_ext['content'].apply(lambda t: TextBlob(t).sentiment.polarity)
df_ext['is_positive'] = (df_ext['sentiment'] > 0.3).astype(int)
df_ext['self_harm']   = df_ext['content'].apply(detect_self_harm)
df_ext['upvote_z'], df_ext['comment_z'] = 0.0, 0.0

# stratified 80/20 split for external
train_ext, hold_ext = train_test_split(
    df_ext, test_size=0.2, stratify=df_ext['label'], random_state=42
)

# === 3) COMBINE FOR TRAINING ===
train_comb = pd.concat([orig_train, train_ext], ignore_index=True)

# features order
FEATURES = [
    'sentiment','is_positive','self_harm','upvote_z','comment_z'
] + temp_feats

# numeric matrix
X_num_train = train_comb[FEATURES].values
y_train     = train_comb['label'].values

# prepare external hold-out numeric
X_num_hold_ext = hold_ext[FEATURES].values
y_hold_ext     = hold_ext['label'].values

# === 4) SENTENCE-BERT EMBEDDINGS ===
print("Computing sentence embeddings (this may take a minute)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# train texts
texts_train = train_comb['content'].tolist()
X_emb_train = embedder.encode(texts_train, show_progress_bar=True)

# hold-out texts
texts_hold_ext = hold_ext['content'].tolist()
X_emb_hold_ext = embedder.encode(texts_hold_ext, show_progress_bar=True)

# combine embed + numeric
X_train_full  = np.hstack([X_emb_train,  X_num_train])
X_hold_ext_full = np.hstack([X_emb_hold_ext, X_num_hold_ext])

# === 5) EVALUATE BASELINE CV ===
hgb = HistGradientBoostingClassifier(
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    l2_regularization=1.0,
    random_state=42
)
cv_scores = cross_val_score(hgb, X_train_full, y_train, cv=5, scoring="f1")
print(f"HGB+Embeddings CV F1: {cv_scores.mean():.3f}")

# === 6) FINAL TRAIN & HOLD-OUT EVAL ===
hgb.fit(X_train_full, y_train)
print("\nHGB+Emb on external hold-out:")
print(classification_report(y_hold_ext, hgb.predict(X_hold_ext_full), digits=3))

# === 7) SAVE ARTIFACTS ===
joblib.dump(embedder, "embedder_sb2.joblib")
joblib.dump(hgb,      "hgb_emb_final.joblib")
print("Saved: embedder_sb2.joblib  &  hgb_emb_final.joblib")
