FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src ./src

RUN python -m pip install --upgrade pip \
    && pip install .

CMD ["bash"]