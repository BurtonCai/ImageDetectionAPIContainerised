
# State python base image to use. 
FROM python:3.7-slim

#Install and update relevant libraries. 

RUN apt-get update -y && apt-get install -y python3-pip python3-dev python3-opencv libsm6 libxext6 libxrender-dev 


WORKDIR /server
COPY iWebLens_server.py /server 
COPY requirements.txt requirements.txt
COPY . /server

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "/server/iWebLens_server.py"]



