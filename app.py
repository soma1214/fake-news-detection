import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load models and vectorizer
@st.cache_data()
def load_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('nb_model.pkl', 'rb') as f:
        fake_news_detection = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        logistic = pickle.load(f)
    return vectorizer, fake_news_detection, logistic

vectorizer, fake_news_detection, logistic = load_models()

ps = PorterStemmer()

st.title("Fake News Detection")
st.write("Enter news article, select model, and click Predict.")

user_input = st.text_area("Enter News Article", height=200)
model_choice = st.selectbox("Select Model", ['Naive Bayes', 'Logistic Regression'])

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter text")
    else:
        # Preprocessing
        news = re.sub('[^a-zA-Z]', ' ', user_input)
        news = news.lower().split()
        news = [ps.stem(word) for word in news if word not in set(stopwords.words('english'))]
        processed_news = ' '.join(news)

        # Vectorize
        vectorized_news = vectorizer.transform([processed_news])

        # Prediction based on selected model
        if model_choice == 'Naive Bayes':
            prediction = fake_news_detection.predict(vectorized_news)[0]
        else:
            prediction = logistic.predict(vectorized_news)[0]

        label_map = {0: 'Fake News', 1: 'Real News'}
        result = label_map[prediction]

        st.subheader(f"{model_choice} Prediction:")
        st.write(result)