FROM python:3.8-slim

RUN apt-get update
RUN apt-get install -y git

RUN mkdir -p home/norte/benchmark_ts
COPY data home/norte/benchmark_ts/

RUN pip install -r home/benchmark_ts/requirements.txt
