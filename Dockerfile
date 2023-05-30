FROM python:3.9-slim-buster

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "myapp:app", "--host", "0.0.0.0", "--port", "8000"]
