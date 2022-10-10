## BUILD STAGE ##
FROM ubuntu:20.04 as builder

RUN mkdir /tools

RUN apt-get update --fix-missing \
  && apt-get install -y tabix \
  && apt-get install -y wget \
  && apt-get install -y python3-pip \
  && apt-get install -y libbz2-dev \
  && apt-get install -y liblzma-dev \
  && apt-get install -y vim

COPY validator* /tools/
COPY utils /tools/utils
COPY requirements.txt /tools

RUN pip3 install -r /tools/requirements.txt
RUN chmod 777 /tools/validator*
ENV PATH=/tools:$PATH