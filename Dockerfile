FROM python:3.9.6-slim

# Install poetry package manager
RUN pip install poetry

# Set application directory
ENV APP_DIR /rl-catch-example
WORKDIR ${APP_DIR}

# Install all python dependencies
COPY pyproject.toml .
RUN poetry config virtualenvs.create false && poetry install --no-dev

COPY . ${APP_DIR}

ENV TRAIN_EPOCHS "1000"
ENV TRAIN_EPSILON "0.1"
ENV TRAIN_MAX_MEMORY "500"
ENV TRAIN_HIDDEN_SIZE "100"
ENV TRAIN HIDDEN_LAYERS "2"
ENV TRAIN_BATCH_SIZE "50"
ENV TRAIN_GRID_SIZE "10"  
ENV RUN_GAME_ITERATIONS "10"

CMD python src/train.py && python src/run.py
