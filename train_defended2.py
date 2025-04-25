#!/usr/bin/env python3
"""
Improved mental health assessment model training pipeline with:
- Advanced embedding model (MPNet)
- Enhanced feature engineering
- Hyperparameter optimization
- Model ensemble
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tqdm import tqdm
import string
from collections import Counter
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# === SETTINGS ===
ORIG_PATH = "/Users/ojaswijoshi/Desktop/DSCI 441(Stat & ML)/Mental-health-assessment/reddit_multi_temporal.csv"
EXT_PATH = "/Users/ojaswijoshi/Downloads/depression_dataset_reddit_cleaned.csv"

# --- Temporal features (must match your CSV) ---
temp_feats = [
    'posts_last_3d', 'avg_sent_last_3d',
    'posts_last_7d', 'avg_sent_last_7d',
    'posts_last_14d', 'avg_sent_last_14d',
    'posts_last_30d', 'avg_sent_last_30d',
    'posts_last_365d', 'avg_sent_last_365d',
    'days_since_prev'
]

# --- Self-harm lexicon (improved with more expressions) ---
self_harm_terms = [
    r"suicid", r"kill myself", r"end my life", r"hurt myself", r"die by suicide",
    r"take my own life", r"jump off", r"no reason to live", r"slit my wrist",
    r"overdose", r"cut myself", r"self[- ]harm", r"don't want to live",
    r"ending it all", r"better off dead", r"can'?t go on", r"want to die",
    r"tired of living", r"hate myself", r"worthless", r"give up", r"hopeless"
]

# --- Emotion lexicons ---
emotion_lexicons = {
    'anxiety': ['worry', 'anxious', 'nervous', 'afraid', 'scared', 'panic', 'fear', 'stress', 'dread', 'phobia'],
    'sadness': ['sad', 'unhappy', 'miserable', 'depressed', 'heartbroken', 'grief', 'sorrow', 'crying', 'tears', 'despair'],
    'anger': ['angry', 'mad', 'frustrated', 'annoyed', 'irritated', 'furious', 'rage', 'hate', 'resent', 'bitter'],
    'loneliness': ['alone', 'lonely', 'isolated', 'abandoned', 'rejected', 'unwanted', 'unloved', 'empty', 'disconnected', 'solitary']
}

# === HELPER FUNCTIONS ===
def detect_self_harm(text: str) -> int:
    t = text.lower()
    return int(any(re.search(term, t) for term in self_harm_terms))

def create_text_features(text):
    """Extract advanced text features."""
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    # Basic features
    word_count = len(words)
    char_count = len(text)
    avg_word_len = char_count / max(word_count, 1)
    
    # Lexical diversity (unique words / total words)
    unique_words = len(set(words))
    diversity = unique_words / max(word_count, 1)
    
    # Punctuation percentage
    punct_count = sum(1 for char in text if char in string.punctuation)
    punct_percent = punct_count / max(char_count, 1)
    
    # Question and exclamation count
    question_count = text.count('?')
    exclamation_count = text.count('!')
    
    # First-person pronoun count (potential signal for self-focus)
    first_person = sum(1 for word in words if word.lower() in ['i', 'me', 'my', 'mine', 'myself'])
    first_person_ratio = first_person / max(word_count, 1)
    
    # Emotion detection using lexicons
    emotion_scores = {}
    for emotion, terms in emotion_lexicons.items():
        count = sum(1 for word in words if word in terms)
        emotion_scores[emotion] = count / max(word_count, 1)
    
    # Negative word ratio
    negative_words = ['not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor', "don't", "can't", "won't"]
    negative_count = sum(1 for word in words if word in negative_words)
    negative_ratio = negative_count / max(word_count, 1)
    
    # Capital letters percentage (may indicate shouting/emphasis)
    caps_count = sum(1 for char in text if char.isupper())
    caps_percent = caps_count / max(char_count, 1)
    
    # Return a dictionary of features
    return {
        'word_count': word_count,
        'avg_word_len': avg_word_len,
        'lexical_diversity': diversity,
        'punct_percent': punct_percent,
        'question_count': question_count,
        'exclamation_count': exclamation_count,
        'first_person_ratio': first_person_ratio,
        'anxiety_score': emotion_scores['anxiety'],
        'sadness_score': emotion_scores['sadness'],
        'anger_score': emotion_scores['anger'],
        'loneliness_score': emotion_scores['loneliness'],
        'negative_ratio': negative_ratio,
        'caps_percent': caps_percent
    }

print("=== Starting Mental Health Risk Model Training Pipeline ===")

# === 1) LOAD & PREP ORIGINAL DATA ===
print("Loading original dataset...")
df_orig = pd.read_csv(ORIG_PATH, parse_dates=['created_dt'])
df_orig.dropna(subset=['content'], inplace=True)
df_orig['content'] = df_orig['content'].str.lower()
df_orig[temp_feats] = df_orig[temp_feats].fillna(0)

# Numeric features on orig
print("Calculating base features for original dataset...")
df_orig['sentiment'] = df_orig['content'].apply(lambda t: TextBlob(t).sentiment.polarity)
df_orig['subjectivity'] = df_orig['content'].apply(lambda t: TextBlob(t).sentiment.subjectivity)
df_orig['is_positive'] = (df_orig['sentiment'] > 0.3).astype(int)
df_orig['self_harm'] = df_orig['content'].apply(detect_self_harm)
# Label = explicit self-harm only
df_orig['label'] = df_orig['self_harm']
df_orig['upvote_z'] = (df_orig['upvotes'] - df_orig['upvotes'].mean()) / df_orig['upvotes'].std()
df_orig['comment_z'] = (df_orig['comments'] - df_orig['comments'].mean()) / df_orig['comments'].std()

# Split orig → 80/20 temporal
cut = int(0.8 * len(df_orig))
orig_train = df_orig.iloc[:cut].reset_index(drop=True)
orig_hold = df_orig.iloc[cut:].reset_index(drop=True)

# === 2) LOAD & PREP EXTERNAL DATA ===
print("Loading external dataset...")
df_ext = pd.read_csv(EXT_PATH)
df_ext.rename(columns={'clean_text': 'content', 'is_depression': 'label'}, inplace=True)
df_ext['content'] = df_ext['content'].str.lower()
df_ext[temp_feats] = 0.0
df_ext['sentiment'] = df_ext['content'].apply(lambda t: TextBlob(t).sentiment.polarity)
df_ext['subjectivity'] = df_ext['content'].apply(lambda t: TextBlob(t).sentiment.subjectivity)
df_ext['is_positive'] = (df_ext['sentiment'] > 0.3).astype(int)
df_ext['self_harm'] = df_ext['content'].apply(detect_self_harm)
df_ext['upvote_z'], df_ext['comment_z'] = 0.0, 0.0

# Stratified 80/20 split for external
train_ext, hold_ext = train_test_split(
    df_ext, test_size=0.2, stratify=df_ext['label'], random_state=42
)

# === 3) ADVANCED FEATURE ENGINEERING ===
print("Extracting advanced text features...")

# Extract advanced text features
def extract_and_add_features(df):
    # Extract features for all texts
    feature_dicts = []
    for text in tqdm(df['content'], desc="Extracting text features"):
        feature_dicts.append(create_text_features(text))
    
    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(feature_dicts)
    
    # Combine with original DataFrame
    result = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    return result

orig_train = extract_and_add_features(orig_train)
train_ext = extract_and_add_features(train_ext)
hold_ext = extract_and_add_features(hold_ext)

# === 4) COMBINE FOR TRAINING ===
print("Combining datasets for training...")
train_comb = pd.concat([orig_train, train_ext], ignore_index=True)

# Expanded feature list
BASE_FEATURES = [
    'sentiment', 'subjectivity', 'is_positive', 'self_harm',
    'upvote_z', 'comment_z'
] + temp_feats

TEXT_FEATURES = [
    'word_count', 'avg_word_len', 'lexical_diversity', 'punct_percent',
    'question_count', 'exclamation_count', 'first_person_ratio',
    'anxiety_score', 'sadness_score', 'anger_score', 'loneliness_score',
    'negative_ratio', 'caps_percent'
]

ALL_FEATURES = BASE_FEATURES + TEXT_FEATURES

# Numeric matrix
X_num_train = train_comb[ALL_FEATURES].values
y_train = train_comb['label'].values

# Prepare external hold-out numeric
X_num_hold_ext = hold_ext[ALL_FEATURES].values
y_hold_ext = hold_ext['label'].values

# === 5) IMPROVED SENTENCE-BERT EMBEDDINGS ===
print("Computing improved sentence embeddings with MPNet (this may take a few minutes)...")
embedder = SentenceTransformer("all-mpnet-base-v2")  # Upgraded model

# Train texts
texts_train = train_comb['content'].tolist()
X_emb_train = embedder.encode(texts_train, show_progress_bar=True)

# Hold-out texts
texts_hold_ext = hold_ext['content'].tolist()
X_emb_hold_ext = embedder.encode(texts_hold_ext, show_progress_bar=True)

# Combine embed + numeric
X_train_full = np.hstack([X_emb_train, X_num_train])
X_hold_ext_full = np.hstack([X_emb_hold_ext, X_num_hold_ext])

# === 6) HYPERPARAMETER OPTIMIZATION ===
print("Optimizing model hyperparameters...")

# Define hyperparameter grid
param_grid = {
    'max_iter': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, None],
    'l2_regularization': [0.0, 0.5, 1.0, 2.0]
}

# Use a subset of data for faster grid search
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Initialize the base model
base_hgb = HistGradientBoostingClassifier(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)

# Grid search
grid_search = GridSearchCV(
    base_hgb, param_grid, cv=3, scoring='f1',
    verbose=1, n_jobs=-1
)
grid_search.fit(X_train_sample, y_train_sample)

# Best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# === 7) MODEL ENSEMBLE ===
print("Training ensemble model...")

# Create optimized HGB model with best parameters
hgb = HistGradientBoostingClassifier(
    max_iter=best_params['max_iter'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    l2_regularization=best_params['l2_regularization'],
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)

# Create a complementary model (Random Forest)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Create a voting ensemble
ensemble = VotingClassifier(
    estimators=[
        ('hgb', hgb),
        ('rf', rf)
    ],
    voting='soft'  # Use probability estimates
)

# Train the ensemble
ensemble.fit(X_train_full, y_train)

# === 8) EVALUATE MODELS ===
print("\nEvaluating models on hold-out set...")

# Compare individual models with ensemble
models = {
    'HGB': hgb.fit(X_train_full, y_train),
    'RF': rf.fit(X_train_full, y_train),
    'Ensemble': ensemble
}

for name, model in models.items():
    y_pred = model.predict(X_hold_ext_full)
    f1 = f1_score(y_hold_ext, y_pred)
    print(f"\n{name} Model Results:")
    print(f"F1 Score: {f1:.3f}")
    print(classification_report(y_hold_ext, y_pred, digits=3))

# === 9) SAVE ARTIFACTS ===
print("Saving model artifacts...")

# Save the feature names for reference
feature_names = {
    'base_features': BASE_FEATURES,
    'text_features': TEXT_FEATURES,
    'all_features': ALL_FEATURES
}
joblib.dump(feature_names, "feature_names.joblib")

# Save the embedder
joblib.dump(embedder, "embedder_mpnet.joblib")

# Determine best model based on hold-out performance
best_model_name = max(models, key=lambda m: f1_score(y_hold_ext, models[m].predict(X_hold_ext_full)))
best_model = models[best_model_name]
print(f"Best model is: {best_model_name}")

# Save the best model
joblib.dump(best_model, "best_model.joblib")

# Also save the ensemble for comparison
joblib.dump(ensemble, "ensemble_model.joblib")

print("Training complete! Saved models:")
print("  - embedder_mpnet.joblib")
print("  - best_model.joblib")
print("  - ensemble_model.joblib")
print("  - feature_names.joblib")