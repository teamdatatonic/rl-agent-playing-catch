FROM python:3

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR .
COPY . .

CMD sh train_and_run.sh
