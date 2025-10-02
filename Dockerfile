FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY data/ ./data/

EXPOSE 8000

CMD ["python", "main.py"]