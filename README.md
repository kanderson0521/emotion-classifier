# Emotion Classifier

A web app built with Python and Flask that can generate predictions based on user input. The emotions included are anger, fear, joy, love, sadness, and surprise. The model was trained with Twitter data obtained here:
https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt

Compared Naïve Bayes and SVM results using different vectorization techniques and parameters to find the best model; achieved 87% accuracy with SVM (Linear) using TF-IDF vectorization.
Currently working to create Docker container and deploy.
