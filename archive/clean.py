import pandas as pd
fake=pd.read_csv("archive/Fake.csv")
true=pd.read_csv("archive/True.csv")
fake['label']=0
true['label']=1
print("Fake Sample")
print(fake[['text','label']].head(3))
print("\nTrue Sample")
print(true[['text','label']].head(3))
data=pd.concat([fake,true],ignore_index=True)
print("\nTotal Articles:",len(data))
print("Label Count:")
print(data['label'].value_counts())
import re
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z/s]',' ',text)
    text=text.strip()
    return text
data['text']=data['text'].apply(clean_text)
data['title']=data['title'].apply(clean_text)
print("\n Cleaned test sample:")
print(data['text'][0][:300])
data.to_csv("cleaned_data.csv",index=False)
print("\nCleaned data saved successfully!")
print("Shape:",data.shape)