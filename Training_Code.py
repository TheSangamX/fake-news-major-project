import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

df = pd.read_csv('true.csv') 
df = pd.read_csv('fake.csv') 


labels = df.label
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)



tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)




pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)




pickle.dump(pac, open('model2.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('tfidfvect2.pkl', 'wb'))


