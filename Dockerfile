FROM python:3.8-slim

EXPOSE 5000

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y \ 
  libgomp1\ 
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

COPY . .

CMD python waitress_server.py