FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py ./
COPY web ./web
ENV AURELIUS_TRUST_PROXY=1 AURELIUS_DB=/data/aurelius.db
RUN mkdir -p /data
CMD ["sh", "-c", "python server.py serve --host 0.0.0.0 --port ${PORT:-8000}"]
