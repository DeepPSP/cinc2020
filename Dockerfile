FROM python:3.7.3-slim

## The MAINTAINER instruction sets the Author field of the generated images
LABEL maintainer="wenh06@gmail.com"
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

