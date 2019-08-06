FROM python:3.7:latest
ADD . /
WORKDIR /code
RUN pip install -r requirements.txt
CMD python app.py