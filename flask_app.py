from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import threading
import nest_asyncio
from waitress import serve

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Setup logging for production
logging.basicConfig(level=logging.INFO)

# Google Colab Specific Fix
nest_asyncio.apply()

# Upload dataset manually if running in Google Colab
from google.colab import files
uploaded = files.upload()

# Get file name dynamically
file_path = list(uploaded.keys())[0]
df = pd.read_csv(file_path)
df.fillna("", inplace=True)

# Ensure required columns exist
required_columns = {"text", "song", "artist"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing columns: {required_columns - set(df.columns)}")

# Define moods and their keywords
mood_keywords = {
    "sad": ["cry", "heartbreak", "lonely", "tears", "pain", "lost"],
    "energetic_gym": ["workout", "pump", "power", "strong", "run", "lift"],
    "romantic": ["love", "heart", "kiss", "forever", "romance", "darling"],
    "nostalgic": ["memories", "past", "childhood", "remember", "old days"],
    "energetic_hype": ["party", "dance", "lit", "hype", "jump", "crazy"],
    "dark_moody": ["night", "mystery", "shadow", "dark", "alone", "whisper"]
}

# Mood detection function (fixes partial word issue)
def assign_mood(text):
    text = text.lower()
    for mood, keywords in mood_keywords.items():
        if any(re.search(r"\b" + word + r"\b", text) for word in keywords):
            return mood
    return "unknown"

df["mood"] = df["text"].apply(assign_mood)

# Feature extraction for similarity
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])

# Convert song names to lowercase for case-insensitive search
df["song"] = df["song"].str.lower()

# API: Recommend top 100 songs by name
@app.route("/recommend/song", methods=["GET"])
def recommend_song():
    song_name = request.args.get("name", "").lower()

    if song_name not in df["song"].values:
        return jsonify({"error": "Song not found."})

    song_index = df[df["song"] == song_name].index[0]
    similarities = cosine_similarity(tfidf_matrix[song_index], tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[-101:-1][::-1]  # Top 100 recommendations
    recommendations = df.iloc[similar_indices]["song"].tolist()

    return jsonify({"recommendations": recommendations})

# API: Recommend top 100 songs by artist
@app.route("/recommend/artist", methods=["GET"])
def recommend_by_artist():
    artist_name = request.args.get("artist", "").lower()

    similar_songs = df[df["artist"].str.lower() == artist_name]["song"].tolist()
    return jsonify({"recommendations": similar_songs[:100]}) if similar_songs else jsonify({"error": "No songs found."})

# API: Recommend top 100 songs by mood
@app.route("/recommend/mood", methods=["GET"])
def recommend_by_mood():
    mood = request.args.get("mood", "").lower()

    mood_songs = df[df["mood"] == mood]["song"].tolist()
    return jsonify({"recommendations": mood_songs[:100]}) if mood_songs else jsonify({"error": "No songs found."})

# Function to run Flask app in the background (Google Colab workaround)
def run_flask():
    serve(app, host="0.0.0.0", port=5000)

# Start Flask server in a thread
thread = threading.Thread(target=run_flask)
thread.start()

print("✅ Flask API is running at port 5000. Use ngrok or local tunneling to access it.")
