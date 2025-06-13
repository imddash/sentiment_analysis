import streamlit as st
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.exceptions import NotFittedError

# --- NLTK setup for Streamlit Cloud ---
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

setup_nltk()

# 🚨 Force download of required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')  # ✅ This is the fix
nltk.download('wordnet')

# --- Load NLTK tools ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Emoji pattern
emoji_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')

# Label mapping
label_map = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# --- Preprocessing Function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    emojis = emoji_pattern.findall(text)
    text = re.sub(r'[^a-z0-9 ]', '', text.lower()) + ' ' + ' '.join(emojis).replace('-', '')
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
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
        st.error(f"❌ Error loading model or vectorizer: {e}")
        st.stop()

vectorizer, model = load_model()

# --- Streamlit App UI ---
st.title("🧠 Sentiment Analysis App")
st.write("Analyze the sentiment of a sentence, tweet, or message using a trained machine learning model.")

user_input = st.text_area("✏️ Enter your text here:", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
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
            st.error("⚠️ Model is not fitted. Please check your training script.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
