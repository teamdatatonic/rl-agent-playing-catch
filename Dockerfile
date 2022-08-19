FROM python:3

RUN pip install poetry

COPY pyproject.toml .

RUN poetry config virtualenvs.create false && poetry install --no-dev

WORKDIR .
COPY . .

CMD sh train_and_run.sh
