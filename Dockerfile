FROM python:3.10-slim
 
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 libxcb-shm0 && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
 
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT"]
 
