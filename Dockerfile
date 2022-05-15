FROM python:3.8-slim
COPY . /usr/local/python
EXPOSE 5000
WORKDIR /usr/local/python
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4
CMD python flask_app.py