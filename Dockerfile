FROM python:3.9.11-slim

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install libgomp1

RUN mkdir -p home/norte/
RUN cd home/norte && git clone -b ppereira https://github.com/p-pereira/benchmark.git
COPY data/1_raw home/norte/benchmark/data/

RUN cd home/norte/benchmark/ && pip install -r requirements.txt

CMD cd home/norte/benchmark/data/6_mlflow && mlflow server --default-artifact-root file:///home/norte/benchmark/data/6_mlflow/mlruns/ -p 5005 -h 0.0.0.0
