import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_df=pd.read_csv('C:/Users/harsh/Downloads/test.csv')
    final_features = final_df['text']
    # prediction = clf.predict(final_features)
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # Cleaning the texts
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(final_features)):
        review = re.sub('[^a-zA-Z]', ' ', final_features['text'][i])
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))
    X_new = cv.fit_transform(corpus).toarray()
    output=clf.predict(X_new)
    if request.method == 'POST':
        message = request.form[final_df['text']]
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('index.html', prediction_text='The predicted sentiment is'.format(my_prediction))


if __name__ == "__main__":
    app.run(debug=True)