FROM python:3.10-slim

RUN apt-get update && apt-get install -y git curl

WORKDIR /workspace

RUN pip install --upgrade pip
RUN pip install torch transformers

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0