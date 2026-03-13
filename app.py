import streamlit as st
import joblib
import re
import numpy as np
model=joblib.load('model.pkl')
vectorizer=joblib.load('vectorizer.pkl')
feature_names=vectorizer.get_feature_names_out()
coefficients=model.coef_[0]
word_importance=dict(zip(feature_names,coefficients))
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z/s]',' ',text)
    text=text.strip()
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
st.set_page_config(page_title="TruthLens", page_icon="🔍")
st.title("TruthLens - Fake News Detector")
st.write("Paste any news article below and I will tell you if it is fake or real.")
article = st.text_area("Enter news article here:", height=200)
if st.button("Analyze"):
    if article.strip() == "":
        st.warning("Please enter an article first!")
    else:
        cleaned = clean_text(article)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        st.divider()

        if prediction == 0:
            st.error("FAKE NEWS DETECTED!")
            st.metric("Confidence", f"{probability[0]*100:.1f}%")
        else:
            st.success("REAL NEWS")
            st.metric("Confidence", f"{probability[1]*100:.1f}%")

        st.divider()
        st.subheader("Why did the AI decide this?")
        top_words = get_top_words(article)

        if top_words:
            for word, score in top_words:
                if score < 0:
                    st.write(f"**{word}** → fake news signal (score: {score:.2f})")
                else:
                    st.write(f"**{word}** → real news signal (score: {score:.2f})")
        else:
            st.write("No significant words found.")