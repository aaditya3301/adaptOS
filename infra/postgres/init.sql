-- Runs once, on the first start of an empty Postgres volume.
-- "adaptos" (app data) is created by POSTGRES_DB; MLflow gets its own database.
CREATE DATABASE mlflow;
