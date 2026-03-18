import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from xgboost import XGBClassifier
import joblib
data=pd.read_csv("cleaned_data.csv")
data['content']=data['title']+' '+data['text']
data=data.dropna(subset=['content'])
vectorizer=joblib.load('vectorizer.pkl')
x=vectorizer.transform(data['content'])
y=data['label']
print("Data loaded and vectorized!")
print("X shape:",x.shape) 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Training XGBoost model...")
print("This may take 2-3 minutes - XGBoost is more complex than Logistic Regression!")
xgb_model=XGBClassifier(n_estimators=100,random_state=42)
xgb_model.fit(x_train,y_train)
y_pred=xgb_model.predict(x_test)
xgb_accuracy=accuracy_score(y_test,y_pred)
print("\n--- RESULTS COMPARISON ---")
print(f"Logistic Regression Accuracy: 98.76%")
print(f"XGBoost Accuracy:             {xgb_accuracy*100:.2f}%")
print("\nDetailed XGBoost Report:")
print(classification_report(y_test,y_pred))
joblib.dump(xgb_model,'xgb_model.pkl')
print("XGBoost model saved!")