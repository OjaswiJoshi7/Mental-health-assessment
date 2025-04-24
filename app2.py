from flask import Flask, render_template, request
import joblib, re
import numpy as np
from textblob import TextBlob

app = Flask(__name__)

# 1) Load your retrained artifacts
embedder = joblib.load("embedder_sb2.joblib")       # SentenceTransformer
hgb_model = joblib.load("hgb_emb_final.joblib")     # HistGradientBoostingClassifier

# 2) Self-harm lexicon (must match training)
self_harm_terms = [
    r"suicid", r"kill myself", r"end my life", r"hurt myself", r"die by suicide",
    r"take my own life", r"jump off", r"no reason to live", r"slit my wrist",
    r"overdose", r"cut myself", r"self[- ]harm", r"don't want to live"
]
def detect_self_harm(text: str) -> int:
    txt = text.lower()
    return int(any(re.search(term, txt) for term in self_harm_terms))

# 3) Rolling-window feature count (11 zeros by default)
temp_feats = [
    'posts_last_3d','avg_sent_last_3d',
    'posts_last_7d','avg_sent_last_7d',
    'posts_last_14d','avg_sent_last_14d',
    'posts_last_30d','avg_sent_last_30d',
    'posts_last_365d','avg_sent_last_365d',
    'days_since_prev'
]

@app.route("/", methods=["GET","POST"])
def index():
    risk_prob = result = None

    if request.method == "POST":
        user_text = request.form["user_text"].strip()
        txt_low = user_text.lower()

        # ——— A) Numeric features ———
        sentiment   = TextBlob(txt_low).sentiment.polarity
        is_positive = int(sentiment > 0.3)
        self_harm   = detect_self_harm(txt_low)
        upvote_z    = 0.0
        comment_z   = 0.0
        rolling     = [0.0] * len(temp_feats)   # eleven zeros

        num_feats = [
            sentiment,
            is_positive,
            self_harm,
            upvote_z,
            comment_z
        ] + rolling                              # total = 5 + 11 = 16 dims

        # ——— B) SBERT embedding (384 dims) ———
        emb = embedder.encode([txt_low])[0]      # shape (384,)

        # ——— C) Build & reshape the feature vector ———
        X = np.concatenate([emb, np.array(num_feats)])   # shape (400,)
        X = X.reshape(1, -1)                             # shape (1, 400)

        # sanity‐check we have the right number of features
        if X.shape[1] != hgb_model.n_features_in_:
            raise ValueError(
                f"Model was trained on {hgb_model.n_features_in_} features, "
                f"but received {X.shape[1]}."
            )

        # ——— D) Predict risk ———
        prob = hgb_model.predict_proba(X)[0, 1]
        risk_prob = f"{prob*100:.1f}%"
        result    = "AT RISK" if prob > 0.5 else "NOT AT RISK"

    return render_template("index.html", risk_prob=risk_prob, result=result)

if __name__ == "__main__":
    app.run(debug=True)
