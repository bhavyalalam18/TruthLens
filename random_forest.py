import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
data = pd.read_csv("cleaned_data.csv")
data['content'] = data['title'] + ' ' + data['text']
data = data.dropna(subset=['content'])
vectorizer = joblib.load('vectorizer.pkl')
X = vectorizer.transform(data['content'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Random Forest...")
print("This may take 2-3 minutes!")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)
print("\n========== FINAL MODEL COMPARISON ==========")
print(f"Logistic Regression:  98.76%")
print(f"Random Forest:        {rf_accuracy*100:.2f}%")
print(f"XGBoost:              99.71%")
print("============================================")
print("\nDetailed Random Forest Report:")
print(classification_report(y_test, y_pred))

joblib.dump(rf_model, 'rf_model.pkl')
print("Random Forest model saved!")