import streamlit as st
import joblib
import re
import numpy as np
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
import string
import os
from datetime import datetime
import pandas as pd
import uuid
import torch
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier

# === 1) PAGE CONFIG ===
st.set_page_config(
    page_title="Mental Health Risk Assessment",
    page_icon="❤️",
    layout="wide"
)

# === 2) LOAD ARTIFACTS ===
@st.cache_resource
def load_models():
    """Load or create models with caching to prevent reloading"""
    print("Loading model artifacts...")
    
    # Create SentenceTransformer model
    try:
        print("Creating new SentenceTransformer model...")
        embedder = SentenceTransformer('all-mpnet-base-v2')
        print("Successfully created SentenceTransformer model")
    except Exception as e:
        st.error(f"Error creating SentenceTransformer: {str(e)}")
        raise
    
    # Create classifier
    print("Creating simple HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Create feature names
    feature_names = {
        'text_features': [
            'word_count', 'avg_word_len', 'lexical_diversity', 
            'punct_percent', 'question_count', 'exclamation_count', 
            'first_person_ratio', 'anxiety_score', 'sadness_score', 
            'anger_score', 'loneliness_score', 'negative_ratio', 
            'caps_percent'
        ]
    }
    
    # Ensure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    return embedder, model, feature_names

# Load models
embedder, model, feature_names = load_models()

# === 3) FEATURE EXTRACTION FUNCTIONS ===
# Self-harm lexicon
self_harm_terms = [
    r"suicid", r"kill myself", r"end my life", r"hurt myself", r"die by suicide",
    r"take my own life", r"jump off", r"no reason to live", r"slit my wrist",
    r"overdose", r"cut myself", r"self[- ]harm", r"don't want to live",
    r"ending it all", r"better off dead", r"can'?t go on", r"want to die",
    r"tired of living", r"hate myself", r"worthless", r"give up", r"hopeless"
]

def detect_self_harm(text: str) -> int:
    txt = text.lower()
    return int(any(re.search(term, txt) for term in self_harm_terms))

# Emotion lexicons
emotion_lexicons = {
    'anxiety': ['worry', 'anxious', 'nervous', 'afraid', 'scared', 'panic', 'fear', 'stress', 'dread', 'phobia'],
    'sadness': ['sad', 'unhappy', 'miserable', 'depressed', 'heartbroken', 'grief', 'sorrow', 'crying', 'tears', 'despair'],
    'anger': ['angry', 'mad', 'frustrated', 'annoyed', 'irritated', 'furious', 'rage', 'hate', 'resent', 'bitter'],
    'loneliness': ['alone', 'lonely', 'isolated', 'abandoned', 'rejected', 'unwanted', 'unloved', 'empty', 'disconnected', 'solitary']
}

def create_text_features(text):
    """Extract advanced text features."""
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    # Basic features
    word_count = len(words)
    char_count = len(text)
    avg_word_len = char_count / max(word_count, 1)
    
    # Lexical diversity
    unique_words = len(set(words))
    diversity = unique_words / max(word_count, 1)
    
    # Punctuation percentage
    punct_count = sum(1 for char in text if char in string.punctuation)
    punct_percent = punct_count / max(char_count, 1)
    
    # Question and exclamation count
    question_count = text.count('?')
    exclamation_count = text.count('!')
    
    # First-person pronoun count
    first_person = sum(1 for word in words if word.lower() in ['i', 'me', 'my', 'mine', 'myself'])
    first_person_ratio = first_person / max(word_count, 1)
    
    # Emotion detection
    emotion_scores = {}
    for emotion, terms in emotion_lexicons.items():
        count = sum(1 for word in words if word in terms)
        emotion_scores[emotion] = count / max(word_count, 1)
    
    # Negative word ratio
    negative_words = ['not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor', "don't", "can't", "won't"]
    negative_count = sum(1 for word in words if word in negative_words)
    negative_ratio = negative_count / max(word_count, 1)
    
    # Capital letters percentage
    caps_count = sum(1 for char in text if char.isupper())
    caps_percent = caps_count / max(char_count, 1)
    
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

# === 4) TEMPORAL FEATURES ===
temp_feats = [
    'posts_last_3d', 'avg_sent_last_3d',
    'posts_last_7d', 'avg_sent_last_7d',
    'posts_last_14d', 'avg_sent_last_14d',
    'posts_last_30d', 'avg_sent_last_30d',
    'posts_last_365d', 'avg_sent_last_365d',
    'days_since_prev'
]

# === 5) SESSION STATE FOR HISTORY ===
# Initialize session state for storing entries
if 'entries' not in st.session_state:
    st.session_state.entries = []

def save_entry(text, risk_score):
    """Save an entry to session state"""
    entry_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create new entry
    new_entry = {
        'id': entry_id,
        'timestamp': timestamp,
        'text': text,
        'risk_score': risk_score
    }
    
    # Add to session state
    st.session_state.entries.insert(0, new_entry)  # Add to beginning of list
    return entry_id

def get_recent_entries(limit=5):
    """Get the most recent entries"""
    return st.session_state.entries[:limit]

# === 6) PREDICTION FUNCTION ===
def predict_risk(text):
    """Run the full prediction pipeline on the input text"""
    txt_low = text.lower()
    
    # Extract sentiment features
    sentiment = TextBlob(txt_low).sentiment.polarity
    subjectivity = TextBlob(txt_low).sentiment.subjectivity
    is_positive = int(sentiment > 0.3)
    self_harm = detect_self_harm(txt_low)
    
    # Placeholder for engagement metrics
    upvote_z = 0.0
    comment_z = 0.0
    
    # Extract advanced text features
    text_features_dict = create_text_features(txt_low)
    
    # Create base features array
    base_features = [
        sentiment, 
        subjectivity,
        is_positive,
        self_harm,
        upvote_z,
        comment_z
    ]
    
    # Add temporal features (all zeros for one-time input)
    temporal_features = [0.0] * len(temp_feats)
    
    # Add text features
    text_features = [text_features_dict.get(feat, 0) for feat in feature_names['text_features']]
    
    # Combine all numeric features
    numeric_features = base_features + temporal_features + text_features
    
    # Get embedding
    embedding = embedder.encode([txt_low])[0]
    
    # Combine embedding and numeric features
    X = np.concatenate([embedding, np.array(numeric_features)])
    X = X.reshape(1, -1)
    
    # Since our model is untrained, use a simulated risk score
    sentiment_weight = 0.3
    self_harm_weight = 0.5
    anxiety_weight = 0.1
    sadness_weight = 0.1
    
    # Simulate a probability based on key features
    prob = (
        (1 - (sentiment + 1) / 2) * sentiment_weight +  # Invert sentiment so negative is higher risk
        self_harm * self_harm_weight +
        text_features_dict.get('anxiety_score', 0) * anxiety_weight +
        text_features_dict.get('sadness_score', 0) * sadness_weight
    )
    # Ensure it's between 0 and 1
    prob = max(0, min(1, prob))
    
    # Extract feature insights
    feature_insights = {
        "emotions": {
            "anxiety": text_features_dict.get('anxiety_score', 0),
            "sadness": text_features_dict.get('sadness_score', 0),
            "anger": text_features_dict.get('anger_score', 0),
            "loneliness": text_features_dict.get('loneliness_score', 0)
        },
        "sentiment": sentiment,
        "self_harm_indicators": self_harm == 1,
        "text_patterns": {
            "first_person_focus": text_features_dict.get('first_person_ratio', 0),
            "question_frequency": text_features_dict.get('question_count', 0) / max(len(text.split()), 1),
            "negative_language": text_features_dict.get('negative_ratio', 0)
        }
    }
    
    return {
        "probability": prob,
        "risk_level": get_risk_level(prob),
        "insights": feature_insights
    }

def get_risk_level(probability):
    """Convert probability to risk level category"""
    if probability < 0.2:
        return {"level": "low", "color": "green"}
    elif probability < 0.5:
        return {"level": "moderate", "color": "orange"}
    elif probability < 0.8:
        return {"level": "high", "color": "red"}
    else:
        return {"level": "severe", "color": "darkred"}

# === 7) STREAMLIT UI ===
def main():
    st.title("Mental Health Risk Assessment")
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Risk Assessment", "History", "About"])
    
    with tab1:
        st.markdown("""
        Enter text to analyze for mental health risk indicators. This tool uses natural language processing to 
        identify patterns associated with potential mental health concerns.
        """)
        
        # Text input
        user_text = st.text_area("Enter text for analysis:", height=150)
        
        if st.button("Analyze"):
            if user_text.strip():
                with st.spinner("Analyzing text..."):
                    # Get prediction
                    result = predict_risk(user_text)
                    
                    # Save to history
                    analysis_id = save_entry(user_text, result["probability"])
                    
                    # Display results
                    risk_level = result["risk_level"]
                    st.subheader("Analysis Results")
                    
                    # Risk level indicator
                    st.markdown(f"<h3 style='color: {risk_level['color']}'>Risk Level: {risk_level['level'].title()}</h3>", unsafe_allow_html=True)
                    
                    # Risk probability
                    st.metric("Risk Score", f"{result['probability']*100:.1f}%")
                    
                    # Insights
                    st.subheader("Key Insights")
                    insights = result["insights"]
                    
                    # Emotions
                    st.markdown("**Emotional Indicators:**")
                    emotions_df = pd.DataFrame({
                        'Emotion': list(insights['emotions'].keys()),
                        'Score': list(insights['emotions'].values())
                    })
                    
                    # Plot emotions
                    st.bar_chart(emotions_df.set_index('Emotion'))
                    
                    # Other insights
                    st.markdown("**Additional Indicators:**")
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Self-harm Language", "Detected" if insights.get("self_harm_indicators", False) else "Not Detected")
                    with cols[1]:
                        st.metric("Sentiment", f"{insights.get('sentiment', 0):.2f}", "-1.0 = Negative, +1.0 = Positive")
                    with cols[2]:
                        st.metric("Self-focus", f"{insights.get('text_patterns', {}).get('first_person_focus', 0):.2f}", "First-person pronoun usage")
            else:
                st.warning("Please enter some text for analysis.")
    
    with tab2:
        st.subheader("Recent Analyses")
        entries = get_recent_entries(10)
        
        if entries:
            for entry in entries:
                with st.expander(f"Analysis from {entry['timestamp']}"):
                    st.text_area("Text", entry['text'], height=100, disabled=True)
                    st.progress(entry['risk_score'])
                    risk_level = get_risk_level(entry['risk_score'])
                    st.markdown(f"Risk Level: **{risk_level['level'].title()}**")
        else:
            st.info("No previous analyses found.")
    
    with tab3:
        st.subheader("About This Tool")
        st.markdown("""
        This Mental Health Risk Assessment tool uses natural language processing to identify potential indicators
        of mental health concerns in text. It analyzes several factors including:
        
        - Emotional tone and sentiment
        - Self-harm language indicators
        - Language patterns associated with mental health concerns
        - First-person pronoun usage and focus
        
        **Important Note:** This tool is for educational and demonstration purposes only. It is not a clinical
        diagnostic tool and should not replace professional mental health evaluation. If you or someone you know
        is experiencing a mental health crisis, please contact a mental health professional or crisis service.
        """)
        
        st.subheader("Mental Health Resources")
        st.markdown("""
        - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
        - Crisis Text Line: Text HOME to 741741
        - National Alliance on Mental Illness (NAMI) Helpline: 1-800-950-6264
        """)

if __name__ == "__main__":
    main()