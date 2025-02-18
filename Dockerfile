FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /workspace/

RUN apt-get update && apt-get install -y gcc
RUN python3 -m pip install -- pip

ADD source/requirements.txt .

RUN pip install -U pip setuptools

RUN pip install -r requirements.txt --no-cache-dir

COPY source/ /workspace/source/
COPY .env /workspace/.env

WORKDIR /workspace/

ENV PYTHONPATH="${PYTHONPATH}:/workspace/."

EXPOSE 5000

CMD ["python", "source/application.py"]