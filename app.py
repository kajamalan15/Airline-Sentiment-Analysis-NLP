# import streamlit as st
# import joblib
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# # Load the trained model and vectorizer
# model = joblib.load("best_sentiment_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")

# # Ensure NLTK resources are available
# nltk.download('punkt')
# nltk.download('stopwords')

# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     words = word_tokenize(text)
#     words = [word for word in words if word not in stopwords.words('english')]
#     return ' '.join(words)

# def predict_sentiment(text):
#     processed_text = preprocess_text(text)
#     vectorized_text = vectorizer.transform([processed_text])
#     prediction = model.predict(vectorized_text)[0]
    
#     sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
#     return sentiment_labels[prediction]

# # Streamlit UI
# st.title("✈️ Airline Tweet Sentiment Analyzer")
# st.write("Enter a tweet below to analyze its sentiment:")

# user_input = st.text_area("Enter your tweet here:")

# if st.button("Analyze Sentiment"):
#     if user_input:
#         sentiment = predict_sentiment(user_input)
#         st.write(f"### Sentiment: {sentiment}")
#         if sentiment == "Positive":
#             st.success("This tweet has a positive sentiment! 😊")
#         elif sentiment == "Negative":
#             st.error("This tweet has a negative sentiment! 😞")
#         else:
#             st.warning("This tweet is neutral. 😐")
#     else:
#         st.warning("Please enter a tweet to analyze.")

# # Run this script with: streamlit run app.py


import streamlit as st
import joblib
import spacy
from spacy.cli import download

# Ensure spaCy model is downloaded
download("en_core_web_sm")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(words)

# Load model and vectorizer
loaded_model = joblib.load("best_sentiment_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Predict sentiment function
def predict_sentiment_loaded(text, model=loaded_model):
    processed_text = preprocess_text(text)
    vectorized_text = loaded_vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    
    sentiment_map = {
        1: ('😊 Positive', 'green'),
        0: ('😞 Negative', 'red'),
        2: ('😐 Neutral', 'orange')
    }
    
    return sentiment_map[prediction]

# Streamlit UI
st.set_page_config(page_title="Airline Sentiment Analysis", layout="centered")

# Custom CSS for aesthetics
st.markdown("""
    <style>
        .sentiment-result {
            font-size: 18px;  /* Reduced font size */
            font-weight: bold;
            text-align: center;
            padding: 8px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .small-text {
            font-size: 13px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("✈️ Airline Sentiment Analysis")

# Input area with placeholder
text_input = st.text_area("Enter your airline review:", placeholder="e.g., 'Great service, but long delays!'")

# Show live character count
st.markdown(f"<p class='small-text'>Character count: {len(text_input)}</p>", unsafe_allow_html=True)

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    if text_input.strip():
        sentiment, color = predict_sentiment_loaded(text_input)
        st.markdown(f'<div class="sentiment-result" style="background-color:{color}; color:white;">{sentiment}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a review to analyze.")
