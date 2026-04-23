import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

print("Starting Model Training...")

# 1. Data Loading (Kaggle Dataset)
# NOTE FOR EXAM: Humne yeh 'fake_news_dataset.csv' Kaggle se download ki thi server par.
# Yeh file 50-100 MB ki hoti hai isliye hum isey GitHub par upload nahi karte.
df = pd.read_csv('fake_news_dataset.csv') 

# Labels ko test aur train me alag karna
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

print("Dataset Loaded. Converting text to vectors...")

# 2. Vectorization (Word into Numbers)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

print("Vectors Created. Training the Machine Learning Model...")

# 3. Model Training
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

print("Training Done! Saving the brain of the model...")

# 4. Save Model to .pkl Files
# Yehi dono files humari website (Flask) app.py me load hoti hain!
pickle.dump(pac, open('model2.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('tfidfvect2.pkl', 'wb'))

print("All files saved. Ready to run on Website.")
