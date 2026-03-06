import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
data=pd.read_csv("cleaned_data.csv")
print("Data loaded:",data.shape)
print(data.head(3))
data['content']=data['title']+' '+data['text']
data = data.dropna(subset=['content'])
print("After removing empty rows:", len(data))
vectorizer=TfidfVectorizer(max_features=5000,stop_words='english')
x=vectorizer.fit_transform(data['content'])
y=data['label']
print('x shape:',x.shape)
print('y shape:',y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Training articles:",x_train.shape)
print("Testing articles:",x_test.shape)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("\nAccuracy:",accuracy_score(y_test,y_pred))
print("\nDetailed Report:")
print(classification_report(y_test,y_pred))
import joblib
joblib.dump(model,'model.pkl')
joblib.dump(vectorizer,'vectorizer.pkl')
print("\nModel saved successfully")