from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
nltk.download('stopwords')
ps = PorterStemmer()

import gc
model = pickle.load(open('model2.pkl', 'rb'))
gc.collect()
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))
gc.collect()
# Monkey-patch variables for sklearn 1.8 compatibility for models saved in older versions
try:
    tfidfvect._tfidf.idf_ = tfidfvect._tfidf._idf_diag.diagonal()
    tfidfvect._tfidf._n_features_out = len(tfidfvect.vocabulary_)
    tfidfvect.n_features_in_ = len(tfidfvect.vocabulary_)
except Exception:
    pass


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)


@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()
