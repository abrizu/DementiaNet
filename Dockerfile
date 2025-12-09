FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    git wget unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

COPY . .    

CMD ["python", "./functions/main.py"]

