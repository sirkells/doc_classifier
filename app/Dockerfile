FROM python:3.7.4-slim-stretch

ADD requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

ADD . /app/

# Set the working directory to /app
WORKDIR /app/

CMD ["python", "main.py"]
#EXPOSE 0.0.0.0:3001:3001/tcp
