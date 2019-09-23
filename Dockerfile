FROM python:3.7.4-slim-stretch

ADD requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

ADD . /webapp

# Set the working directory to /app
WORKDIR /webapp

CMD ["python", "main.py"]

