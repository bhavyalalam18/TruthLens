import pandas as pd
import joblib
import numpy as np
model=joblib.load('model.pkl')
vectorizer=joblib.load('vectorizer.pkl')
feature_names=vectorizer.get_feature_names_out()
coefficients=model.coef_[0]
word_importance=dict(zip(feature_names,coefficients))
fake_words=sorted(word_importance.items(),key=lambda x:x[1])[:10]
real_words=sorted(word_importance.items(),key=lambda x:x[1], reverse=True)[:10]
print("Top 10 words that indicate FAKE news:")
for word, score in fake_words:
    print(f" {word}: {score:.3f}")
print("\nTop 10 words that indicate REAL news:")
for word, score in real_words:
    print(f" {word}: {score:.3f}")
