# 🧠 Sentiment Analysis Web App

This is a simple and interactive **Sentiment Analysis** web application built using **Streamlit** and **scikit-learn**. It uses a pre-trained machine learning model to predict whether a given piece of text expresses a **positive** or **negative** sentiment.

---

## ✨ Demo

Paste a sentence or review, and the model will classify the sentiment instantly.

> Example:  
> **"I absolutely loved this product!"** → ✅ Positive  
> **"It was a terrible experience."** → ❌ Negative

---

## 📦 Features

- 🧾 User-friendly Streamlit interface for inputting text
- 🔤 Real-time predictions using a logistic regression classifier
- 📁 Pre-trained model and vectorizer included (`.pkl` files)
- 📈 Built using standard NLP techniques (`TfidfVectorizer`)

---

## 🧰 Tech Stack

- Python
- Streamlit
- scikit-learn
- joblib (for saving/loading the model)
- Git & GitHub

---

## 📁 Project Structure

sentiment_analysis/  
├── app.py # Streamlit web app  
├── requirements.txt # Dependencies  
├── model/  
│ ├── sentiment_model.pkl # Trained sentiment classifier  
│ └── vectorizer.pkl # TfidfVectorizer for text preprocessing


---

## ▶️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/imddash/sentiment_analysis.git   
cd sentiment_analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py  
```
ℹ️ Make sure sentiment_model.pkl and vectorizer.pkl are inside the model/ folder.



📄 License  
This project is open-source and available under the MIT License.
