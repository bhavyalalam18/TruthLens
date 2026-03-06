import streamlit as st
import joblib
import re
model=joblib.load('model.pkl')
vectorizer=joblib.load('vectorizer.pkl')
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z/s]',' ',text)
    text=text.strip()
    return text
st.title("TruthLens - Fake News Detector")
st.write("Paste any news article below and I will tell you if it is fake or real.")
article=st.text_area("Enter new articles here:",height=200)
if st.button("Analyze"):
    if article.strip() =="":
        st.warning(" Please enter an article first!")
    else:
        cleaned=clean_text(article)
        vectorized=vectorizer.transform([cleaned])
        prediction=model.predict(vectorized)[0]
        probability=model.predict_proba(vectorized)[0]
        if prediction==0:
            st.error("FAKE NEWS PREDICTED")
            st.write(f"confidence:{probability[0]*100:.1f}%")
        else:
            st.success("REAL NEWS")
            st.write(f"confidence:{probability[0]*100:.1f}%")    