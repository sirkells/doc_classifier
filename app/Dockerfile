FROM python:3.7.2-stretch

ADD . /app/

# Set the working directory to /app
WORKDIR /app/

# Install the dependencies
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

CMD ["python", "main.py"]
EXPOSE 0.0.0.0:3001:3001/tcp 

