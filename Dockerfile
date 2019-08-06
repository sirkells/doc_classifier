FROM ubuntu:18.04


RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./app/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "main.py" ]