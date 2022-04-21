import pickle
from flask import Flask, request, render_template
from nltk import word_tokenize
from nltk import WordNetLemmatizer

# Load in model and tfidf vec
with open('./data/emotion_clf_svm_v1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('./data/tfidf_vec.pkl', 'rb') as tfidf_file:
    tfidf_vec = pickle.load(tfidf_file)
    # Parameters: encoding='latin-1', binary=False, lowercase=True, ngram_range=(1,2), stopwords=stop_words

app = Flask(__name__)


def lemmatize_verbs(text):
    lemmer = WordNetLemmatizer()
    words = word_tokenize(text)
    lem_output = ' '.join([lemmer.lemmatize(w, pos='v') for w in words])
    return lem_output


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('input_text')
    text = [lemmatize_verbs(text)]
    text_tf_idf = tfidf_vec.transform(text)
    if text_tf_idf.sum(axis=1) == 0:
        no_words = "No words found in current model, try with another sentence."
        return render_template('home.html', result=no_words)
    else:
        emotion = model.predict(text_tf_idf)
        return render_template('home.html', result="The predicted emotion is " + str(emotion[0]))


@app.route('/')
def main():
    return render_template('home.html',)


if __name__ == '__main__':
    app.run()