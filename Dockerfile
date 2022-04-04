FROM python:3.8-slim

RUN apt-get update
RUN apt-get install -y git

RUN mkdir -p /home/norte/
RUN cd /home/norte && git clone -b ncosta https://github.com/p-pereira/benchmark.git
ADD data/ /home/norte/benchmark/

RUN cd /home/norte/benchmark/ && pip install -r requirements.txt

CMD cd /home/norte/benchmark/data/6_mlflow && mlflow ui --default-artifact-root file:///home/norte/benchmark/data/6_mlflow/mlruns/