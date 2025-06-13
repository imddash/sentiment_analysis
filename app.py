import streamlit as st
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.exceptions import NotFittedError

# Download NLTK data (safe to run multiple times)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Emoji pattern
emoji_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')

# Label mapping (adjust according to your model)
label_map = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# --- Preprocessing Function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Extract emojis
    emojis = emoji_pattern.findall(text)
    # Clean text
    text = re.sub(r'[^a-z0-9 ]', '', text.lower()) + ' ' + ' '.join(emojis).replace('-', '')
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # POS tagging and lemmatization
    tagged = pos_tag(words)
    lemmas = []
    for word, tag in tagged:
        pos = 'n'
        if tag.startswith('J'):
            pos = 'a'
        elif tag.startswith('V'):
            pos = 'v'
        elif tag.startswith('R'):
            pos = 'r'
        lemmas.append(lemmatizer.lemmatize(word, pos))
    return ' '.join(lemmas)

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model():
    try:
        vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))
        model = joblib.load(os.path.join("model", "sentiment_model.pkl"))
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()

vectorizer, model = load_model()

# --- Streamlit App UI ---
st.title("🧠 Sentiment Analysis Web App")
st.write("Analyze the sentiment of any sentence or social media text using a trained ML model.")

user_input = st.text_area("Enter text here:", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess_text(user_input)
        try:
            vectorized_input = vectorizer.transform([clean_text])
            prediction = model.predict(vectorized_input)[0]
            probabilities = model.predict_proba(vectorized_input)[0]

            st.markdown(f"### 🎯 Predicted Sentiment: **{label_map[prediction]}**")
            st.markdown("#### 📊 Confidence Levels:")
            for idx, prob in enumerate(probabilities):
                st.write(f"{label_map[idx]}: {round(prob * 100, 2)}%")
        except NotFittedError:
            st.error("Model is not fitted. Please check your training code.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
