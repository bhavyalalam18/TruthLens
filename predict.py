import joblib
import re
model=joblib.load('model.pkl')
vectorizer=joblib.load('vectorizer.pkl')
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]',' ',text)
    text=text.strip()
    return text
def predict(article):
    cleaned=clean_text(article)
    vectorized=vectorizer.transform([cleaned])
    prediction=model.predict(vectorized)[0]
    probability=model.predict_proba(vectorized)[0]
    if prediction==0:
        print("Fake News Detected!")
        print(f"Confidence: {probability[0]*100:.1f}%")
    else:
        print("Real News")
        print(f"Confidence : {probability[1]*100:.1f}%")
article1 = "NASA confirms water discovered on mars according to scientists"
article2 = "BOMBSHELL Obama secretly controls the government illuminati exposed"
print("Article 1:")
predict(article1)
print("\nArticle 2:")
predict(article2)