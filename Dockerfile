# Face Search Tool - Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/app/face_db"]

EXPOSE 8000

ENV SESSION_SECRET_KEY=""
ENV JWT_SECRET_KEY=""
ENV ADMIN_PASSWORD=""

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]