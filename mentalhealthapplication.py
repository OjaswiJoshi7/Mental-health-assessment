from flask import Flask, render_template, request, jsonify
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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Make sure templates directory exists
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'), exist_ok=True)

# === 1) SIMPLIFIED MODEL APPROACH ===
logger.info("Initializing model components...")

# Create default feature names
feature_names = {
    'text_features': [
        'word_count', 'avg_word_len', 'lexical_diversity', 
        'punct_percent', 'question_count', 'exclamation_count', 
        'first_person_ratio', 'anxiety_score', 'sadness_score', 
        'anger_score', 'loneliness_score', 'negative_ratio', 
        'caps_percent'
    ]
}

# Simplified embedding approach - create a random vector instead of using SentenceTransformer
def get_simplified_embedding(text):
    """Create a simplified embedding instead of using SentenceTransformer"""
    # Return a random vector of size 768 (similar to what SentenceTransformer would return)
    # In a real app, you'd want to use a proper embedding model
    np.random.seed(hash(text) % 2**32)  # Use text hash as seed for deterministic behavior
    return np.random.normal(0, 1, 768)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# === 2) DEFINE FEATURE EXTRACTION FUNCTIONS ===
# Self-harm lexicon (must match training)
self_harm_terms = [
    r"suicid", r"kill myself", r"end my life", r"hurt myself", r"die by suicide",
    r"take my own life", r"jump off", r"no reason to live", r"slit my wrist",
    r"overdose", r"cut myself", r"self[- ]harm", r"don't want to live",
    r"ending it all", r"better off dead", r"can'?t go on", r"want to die",
    r"tired of living", r"hate myself", r"worthless", r"give up", r"hopeless"
]

def detect_self_harm(text: str) -> int:
    """Detect presence of self-harm indicators in text"""
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
    
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(nltk.corpus.stopwords.words('english'))
    
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

# === 3) TEMPORAL FEATURES (default zeros) ===
temp_feats = [
    'posts_last_3d', 'avg_sent_last_3d',
    'posts_last_7d', 'avg_sent_last_7d',
    'posts_last_14d', 'avg_sent_last_14d',
    'posts_last_30d', 'avg_sent_last_30d',
    'posts_last_365d', 'avg_sent_last_365d',
    'days_since_prev'
]

# === 4) DATABASE SIMULATION (for storing user entries) ===
DATA_FILE = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'user_entries.csv')

def initialize_data_file():
    """Create the CSV file if it doesn't exist"""
    # Define data_file as a local variable first
    data_file = DATA_FILE
    
    try:
        if not os.path.exists(data_file):
            df = pd.DataFrame(columns=['id', 'timestamp', 'text', 'risk_score'])
            df.to_csv(data_file, index=False)
            logger.info(f"Initialized data file at {data_file}")
        else:
            logger.info(f"Data file already exists at {data_file}")
    except Exception as e:
        logger.error(f"Error initializing data file: {str(e)}")
        # Create a backup location in the current directory if /tmp is not accessible
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_entries.csv')
        if not os.path.exists(data_file):
            df = pd.DataFrame(columns=['id', 'timestamp', 'text', 'risk_score'])
            df.to_csv(data_file, index=False)
            logger.info(f"Initialized backup data file at {data_file}")
    
    return data_file

def save_entry(text, risk_score):
    """Save an entry to our CSV 'database'"""
    try:
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load existing data
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
        else:
            df = pd.DataFrame(columns=['id', 'timestamp', 'text', 'risk_score'])
        
        # Add new entry
        new_entry = pd.DataFrame({
            'id': [entry_id],
            'timestamp': [timestamp],
            'text': [text],
            'risk_score': [risk_score]
        })
        
        # Append and save
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        logger.info(f"Saved entry with ID {entry_id}")
        return entry_id
    except Exception as e:
        logger.error(f"Error saving entry: {str(e)}")
        return str(uuid.uuid4())  # Return a dummy ID in case of error

def get_recent_entries(limit=5):
    """Get the most recent entries (for history feature)"""
    try:
        if not os.path.exists(DATA_FILE):
            logger.warning(f"Data file does not exist at {DATA_FILE}")
            return []
        
        df = pd.read_csv(DATA_FILE)
        if df.empty:
            return []
        
        # Sort by timestamp (newest first) and take the top entries
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        recent = df.head(limit)
        
        # Convert to list of dicts for template rendering
        return recent.to_dict('records')
    except Exception as e:
        logger.error(f"Error getting recent entries: {str(e)}")
        return []

# === 5) PREDICTION FUNCTION ===
def predict_risk(text):
    """Run the full prediction pipeline on the input text"""
    try:
        txt_low = text.lower()
        
        # Extract sentiment features
        sentiment = TextBlob(txt_low).sentiment.polarity
        subjectivity = TextBlob(txt_low).sentiment.subjectivity
        is_positive = int(sentiment > 0.3)
        self_harm = detect_self_harm(txt_low)
        
        # Placeholder for engagement metrics (would come from user account in real app)
        upvote_z = 0.0
        comment_z = 0.0
        
        # Extract advanced text features
        text_features_dict = create_text_features(txt_low)
        
        # Create base features array (must match training order)
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
        
        # Add text features (in correct order)
        text_features = [text_features_dict[feat] for feat in feature_names['text_features']]
        
        # Combine all numeric features
        numeric_features = base_features + temporal_features + text_features
        
        # Get embedding using simplified approach
        embedding = get_simplified_embedding(txt_low)
        
        # Combine embedding and numeric features
        X = np.concatenate([embedding, np.array(numeric_features)])
        
        # Since we don't have a real model, use simplified logic to generate a risk score
        # This is just a placeholder for demonstration - in a real app, you'd use the actual model
        # The logic below tries to approximate what the model might do
        
        # Risk increases with self_harm indicators and negative emotions
        base_risk = 0.2  # Start with a baseline risk
        
        # Self-harm indicators have a large impact
        if self_harm == 1:
            base_risk += 0.3
        
        # Negative sentiment increases risk
        if sentiment < 0:
            base_risk += abs(sentiment) * 0.1
        
        # High emotion scores increase risk
        emotion_factor = 0.1 * sum([
            text_features_dict['anxiety_score'],
            text_features_dict['sadness_score'], 
            text_features_dict['anger_score'],
            text_features_dict['loneliness_score']
        ])
        base_risk += emotion_factor
        
        # High negative language increases risk
        base_risk += text_features_dict['negative_ratio'] * 0.1
        
        # Cap at 0.95 for severe cases and ensure minimum of 0.05
        prob = max(0.05, min(0.95, base_risk))
        
        # Extract top contributing features
        feature_insights = {
            "emotions": {
                "anxiety": text_features_dict['anxiety_score'],
                "sadness": text_features_dict['sadness_score'],
                "anger": text_features_dict['anger_score'],
                "loneliness": text_features_dict['loneliness_score']
            },
            "sentiment": sentiment,
            "self_harm_indicators": self_harm == 1,
            "text_patterns": {
                "first_person_focus": text_features_dict['first_person_ratio'],
                "question_frequency": text_features_dict['question_count'] / max(len(text.split()), 1),
                "negative_language": text_features_dict['negative_ratio']
            }
        }
        
        logger.info(f"Prediction complete with risk level {prob}")
        return {
            "probability": prob,
            "risk_level": get_risk_level(prob),
            "insights": feature_insights
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        # Return a fallback response in case of error
        return {
            "probability": 0.1,
            "risk_level": {"level": "low", "color": "green"},
            "insights": {
                "emotions": {"anxiety": 0, "sadness": 0, "anger": 0, "loneliness": 0},
                "sentiment": 0,
                "self_harm_indicators": False,
                "text_patterns": {"first_person_focus": 0, "question_frequency": 0, "negative_language": 0}
            }
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

# === 6) FLASK ROUTES ===
@app.route("/", methods=["GET", "POST"])
def index():
    """Main page route"""
    result = None
    analysis_id = None
    history = get_recent_entries(5)
    
    if request.method == "POST":
        user_text = request.form.get("user_text", "").strip()
        
        if user_text:
            # Get prediction
            result = predict_risk(user_text)
            
            # Save to our "database"
            analysis_id = save_entry(user_text, result["probability"])
            
            # Refresh history
            history = get_recent_entries(5)
    
    return render_template(
        "index_improved.html", 
        result=result, 
        analysis_id=analysis_id,
        history=history
    )

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint for AJAX analysis"""
    try:
        data = request.json
        user_text = data.get("text", "").strip() if data else ""
        
        if not user_text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get prediction
        result = predict_risk(user_text)
        
        # Save entry
        analysis_id = save_entry(user_text, result["probability"])
        
        # Add ID to result
        result["id"] = analysis_id
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": "An error occurred during analysis"}), 500

@app.route("/history")
def history():
    """View history of analyses"""
    entries = get_recent_entries(20)  # Show more in dedicated history page
    return render_template("history.html", entries=entries)

@app.route("/about")
def about():
    """About page with information about the model"""
    return render_template("about.html")

@app.route("/resources")
def resources():
    """Mental health resources page"""
    return render_template("resources.html")

# === 7) INITIALIZE AND RUN ===
if __name__ == "__main__":
    try:
        # Get the data file path and update the global variable
        data_file_path = initialize_data_file()
        # Update global DATA_FILE with the returned path
        globals()['DATA_FILE'] = data_file_path
        
        port = int(os.environ.get("PORT", 5001))
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")