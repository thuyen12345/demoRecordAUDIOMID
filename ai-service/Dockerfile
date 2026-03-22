FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 🔥 FIX ở đây
RUN pip install --upgrade pip 
RUN pip install --no-cache-dir "setuptools==68.2.2" wheel
# 🔥 DÁN Ở ĐÂY
RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117
RUN python -c "import whisper; print('whisper installed ok')"

# # install whisper riêng
# RUN pip install openai-whisper==20231117

# debug file
RUN cat requirements.txt

# install deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p storage/audio storage/temp logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]