FROM python:3

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR .
COPY . .

CMD python src/train.py
CMD python src/run.py