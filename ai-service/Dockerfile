FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# System libs for OpenCV + audio + build tools
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsndfile1 \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*
USER user

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
