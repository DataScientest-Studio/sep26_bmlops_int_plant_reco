FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.11.1 psycopg2-binary==2.9.13
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
