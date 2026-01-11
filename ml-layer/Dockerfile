FROM python:3.10

WORKDIR /app

# Make /app a Python package root
ENV PYTHONPATH=/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.api.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
