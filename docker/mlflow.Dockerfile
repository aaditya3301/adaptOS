FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.18.0 psycopg2-binary==2.9.10

RUN useradd --create-home --uid 1000 mlflow \
    && mkdir -p /mlartifacts && chown mlflow:mlflow /mlartifacts
USER mlflow

EXPOSE 5000
