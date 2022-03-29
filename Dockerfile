FROM python:3.8-slim

RUN apt-get update
RUN apt-get install -y git

RUN mkdir -p home/norte/
RUN cd home/norte
RUN git clone https://github.com/p-pereira/benchmark_ts.git
COPY data home/norte/benchmark_ts/

RUN pip install -r home/benchmark_ts/requirements.txt
