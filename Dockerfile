FROM python:3.7.4

ADD requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install  --no-cache --no-cache-dir --upgrade -r requirements.txt
RUN pip install uwsgi
RUN rm -fr ~/.cache/pip /tmp*
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet


ADD . /webapp

# Set the working directory to /app
WORKDIR /webapp

ENTRYPOINT ["uwsgi"]
CMD ["--socket", "0.0.0.0:8000", "-b", "65535" ,"--wsgi-file", "main.py", "--callable", "app", "--processes", "1", "--threads", "8"]
#CMD ["python", "main.py"]

