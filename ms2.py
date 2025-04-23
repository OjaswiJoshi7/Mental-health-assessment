#!/usr/bin/env python3
"""
1. Retrain on combined original + external data (including temporal/user features)
2. Evaluate on original hold-out
3. Evaluate on external hold-out
4. Threshold tuning, hyperparameter search, and error analysis
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# === SETTINGS ===
ORIG_PATH = "reddit_multi_temporal.csv"   # enriched CSV with temporal features
EXT_PATH  = "/Users/ojaswijoshi/Downloads/depression_dataset_reddit_cleaned.csv"

# === 1) LOAD DATA ===
df_orig = pd.read_csv(ORIG_PATH, parse_dates=['created_dt'])
df_orig.dropna(subset=['content'], inplace=True)
df_orig['content'] = df_orig['content'].str.lower()

# List of your rolling/temporal columns
temp_feats = [
    'posts_last_3d','avg_sent_last_3d',
    'posts_last_7d','avg_sent_last_7d',
    'posts_last_14d','avg_sent_last_14d',
    'posts_last_30d','avg_sent_last_30d',
    'days_since_prev'
]

# **IMPUTE ALL TEMPORAL FEATURES TO 0 WHEREVER MISSING**  
df_orig[temp_feats] = df_orig[temp_feats].fillna(0)

# Load external Kaggle dataset
df_ext = pd.read_csv(EXT_PATH)
df_ext.rename(columns={'clean_text':'content','is_depression':'label'}, inplace=True)
df_ext['content'] = df_ext['content'].str.lower()

# Dummy out engagement & temporal features in the external set
df_ext['upvote_z']  = 0.0
df_ext['comment_z'] = 0.0
for col in temp_feats:
    df_ext[col] = 0.0

# === 2) PREPARE ORIGINAL LABELS & ENGAGEMENT ===
df_orig['sentiment'] = df_orig['content'].apply(lambda t: TextBlob(t).sentiment.polarity)
df_orig['label']     = (df_orig['sentiment'] < -0.3).astype(int)
df_orig['upvote_z']  = (df_orig['upvotes']   - df_orig['upvotes'].mean()) / df_orig['upvotes'].std()
df_orig['comment_z'] = (df_orig['comments']  - df_orig['comments'].mean()) / df_orig['comments'].std()

# === 3) TRAIN/HOLD SPLITS ===
cut        = int(0.8 * len(df_orig))
orig_train = df_orig.iloc[:cut].reset_index(drop=True)
orig_hold  = df_orig.iloc[cut:].reset_index(drop=True)

train_ext, hold_ext = train_test_split(
    df_ext, test_size=0.2,
    stratify=df_ext['label'], random_state=42
)

# === 4) COMBINE FOR RE-TRAINING ===
train_comb = pd.concat([
    orig_train[['content','upvote_z','comment_z','label'] + temp_feats],
    train_ext[['content','upvote_z','comment_z','label'] + temp_feats]
], ignore_index=True)

# **Since we reloaded from CSV, ensure no NaNs in temporal columns**
train_comb[temp_feats] = train_comb[temp_feats].fillna(0)

# Recompute sentiment on the combined set
train_comb['sentiment'] = train_comb['content'] \
    .apply(lambda t: TextBlob(t).sentiment.polarity)

# Anxiety keyword count
anxiety_terms = {'hopeless','worthless','suicidal','alone'}
train_comb['anxiety_count'] = train_comb['content'].apply(
    lambda txt: sum(1 for w in re.findall(r'\b\w+\b', txt) if w in anxiety_terms)
)

# === 5) FIT VECTORIZER, LDA, AND GB ===
stops      = ENGLISH_STOP_WORDS.union({'thing','one'})
vectorizer = CountVectorizer(stop_words=list(stops), min_df=5)
dtm_comb   = vectorizer.fit_transform(train_comb['content'])
lda        = LatentDirichletAllocation(n_components=5, random_state=42)
topics     = lda.fit_transform(dtm_comb)
for i in range(5):
    train_comb[f'topic_{i}'] = topics[:, i]

FEATURES = (
    ['sentiment','upvote_z','comment_z','anxiety_count']
    + temp_feats
    + [f'topic_{i}' for i in range(5)]
)

# Balance the minority class
maj      = train_comb[train_comb.label==0]
min_cls  = train_comb[train_comb.label==1]
min_up   = resample(min_cls, replace=True, n_samples=len(maj), random_state=42)
train_bal = pd.concat([maj, min_up])

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(train_bal[FEATURES], train_bal['label'])

# Save pipeline components
joblib.dump(vectorizer, 'vectorizer_comb.joblib')
joblib.dump(lda,         'lda_comb.joblib')
joblib.dump(gb,          'gb_comb.joblib')

# Helper to add topics/anxiety/sentiment for hold-outs
def featureize(df, vec, lda_model):
    df = df.copy()
    df[temp_feats] = df[temp_feats].fillna(0)          # fill any missing temp features
    df['sentiment']      = df['content'].apply(lambda t: TextBlob(t).sentiment.polarity)
    df['anxiety_count']  = df['content'].apply(
        lambda txt: sum(1 for w in re.findall(r'\b\w+\b', txt) if w in anxiety_terms)
    )
    dtm = vec.transform(df['content'])
    tops = lda_model.transform(dtm)
    for i in range(tops.shape[1]):
        df[f'topic_{i}'] = tops[:, i]
    return df

# === 6) Evaluate on original hold-out ===
hold_o = featureize(orig_hold, vectorizer, lda)
X_o, y_o = hold_o[FEATURES], hold_o['label']
print("\nOriginal hold-out:")
print(classification_report(y_o, gb.predict(X_o), digits=3))

# === 7) Evaluate on external hold-out ===
hold_e = featureize(hold_ext, vectorizer, lda)
X_e, y_e = hold_e[FEATURES], hold_e['label']
print("\nExternal hold-out:")
print(classification_report(y_e, gb.predict(X_e), digits=3))

# === 8) Threshold tuning on external hold-out ===
probs_ext    = gb.predict_proba(X_e)[:,1]
prec_ext, rec_ext, thr_ext = precision_recall_curve(y_e, probs_ext)
plt.figure(figsize=(8,4))
plt.plot(thr_ext, rec_ext[:-1], label='Recall')
plt.plot(thr_ext, prec_ext[:-1], label='Precision')
plt.xlabel("Threshold"); plt.ylabel("Score")
plt.title("External: Precision vs Recall")
plt.legend(); plt.show()

TARGET_REC   = 0.80
valid_thr    = [t for r,t in zip(rec_ext, thr_ext) if r >= TARGET_REC]
best_ext_thr = max(valid_thr) if valid_thr else 0.5
print(f"\nChosen threshold (recall ≥ {TARGET_REC}): {best_ext_thr:.3f}")

tuned_preds = (probs_ext >= best_ext_thr).astype(int)
print("\nExternal @ tuned threshold:")
print(classification_report(y_e, tuned_preds, digits=3))

# === 9) Hyperparameter tuning ===
param_grid = {
    'n_estimators':   [50,100,200],
    'max_depth':      [3,5],
    'learning_rate': [0.01,0.1]
}
grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, scoring='f1', cv=5, n_jobs=-1
)
grid.fit(train_bal[FEATURES], train_bal['label'])
print("\nBest GB params:", grid.best_params_)

gb_tuned = grid.best_estimator_
print("\nTuned on original hold-out:")
print(classification_report(y_o, gb_tuned.predict(X_o), digits=3))
print("\nTuned on external hold-out:")
print(classification_report(y_e, gb_tuned.predict(X_e), digits=3))

# === 10) Error analysis on external hold-out ===
errors = hold_e.copy()
errors['prob'] = probs_ext
errors['pred'] = tuned_preds
mis = errors[errors['pred'] != errors['label']]
print("\nSample misclassifications:")
for _, row in mis.sample(10, random_state=42).iterrows():
    print(f"- {row['label']}→{row['pred']} @{row['prob']:.2f}: {row['content'][:200]}...")
