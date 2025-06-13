import streamlit as st
import re
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# NLTK downloads (do only once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load stopwords and lemmatizer
stopwords_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Emoji regex
emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)')

# Label mapping (based on your training setup)
label_map = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# Preprocessing function
def preprocessing(text):
    if not isinstance(text, str):
        return ""
    text = re.sub('<[^>]*>', '', text)  # Remove HTML
    emojis = emoji_pattern.findall(text)
    text = re.sub('[^a-z0-9 ]', '', text.lower()) + ' ' + ' '.join(emojis).replace('-', '')
    words = text.split()
    words = [word for word in words if word not in stopwords_set]
    tagged_words = nltk.pos_tag(words)
    lemmatized_words = []
    for word, tag in tagged_words:
        pos = 'n'
        if tag.startswith('J'):
            pos = 'a'
        elif tag.startswith('V'):
            pos = 'v'
        elif tag.startswith('R'):
            pos = 'r'
        lemmatized_words.append(lemmatizer.lemmatize(word, pos))
    return " ".join(lemmatized_words)

# Load model and vectorizer
import joblib
import os
vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))
model = joblib.load(os.path.join("model", "sentiment_model.pkl"))

# Streamlit UI
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter a sentence or tweet:")

if st.button("Analyze Sentiment"):
    clean_text = preprocessing(user_input)
    X_input = vectorizer.transform([clean_text])

    # Get prediction and probabilities
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    # Show overall sentiment
    st.markdown(f"### Predicted Sentiment: **{label_map[prediction]}**")

    # Show probability breakdown
    st.markdown("#### Sentiment Confidence:")
    for label, prob in enumerate(probabilities):
        st.write(f"{label_map[label]}: {round(prob * 100, 2)}%")
