import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd

# Load best model (XGBoost) and vectorizer
model = joblib.load('xgb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
lr_model = joblib.load('model.pkl')
feature_names = vectorizer.get_feature_names_out()
lr_coefficients = lr_model.coef_[0]
word_importance = dict(zip(feature_names, lr_coefficients))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = text.strip()
    return text

def get_top_words(article, top_n=5):
    cleaned = clean_text(article)
    words = cleaned.split()
    word_scores = []
    for word in words:
        if word in word_importance:
            word_scores.append((word, word_importance[word]))
    word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)
    return word_scores[:top_n]

# Page config
st.set_page_config(page_title="TruthLens", page_icon="🔍", layout="wide")

# Title
st.title("TruthLens - Fake News Detector")
st.write("Powered by XGBoost — 99.71% accuracy on 44,898 news articles")
st.divider()

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Analyze Article")
    article = st.text_area("Paste your news article here:", height=250)
    analyze_btn = st.button("Analyze", use_container_width=True)

with col2:
    st.subheader("Model Performance")
    perf_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [98.76, 99.68, 99.71]
    })
    st.bar_chart(perf_data.set_index('Model'))

st.divider()

if analyze_btn:
    if article.strip() == "":
        st.warning("Please enter an article first!")
    else:
        cleaned = clean_text(article)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        result_col, words_col = st.columns(2)

        with result_col:
            if prediction == 0:
                st.error("FAKE NEWS DETECTED!")
                st.metric("Confidence", f"{probability[0]*100:.1f}%")
            else:
                st.success("REAL NEWS")
                st.metric("Confidence", f"{probability[1]*100:.1f}%")

        with words_col:
            st.subheader("Why did AI decide this?")
            top_words = get_top_words(article)
            if top_words:
                for word, score in top_words:
                    if score < 0:
                        st.write(f"**{word}** → fake signal ({score:.2f})")
                    else:
                        st.write(f"**{word}** → real signal ({score:.2f})")