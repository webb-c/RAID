FROM python:3.10

# Mount volume
RUN mkdir /volume
WORKDIR /volume
ADD ./requirements.txt /volume

# Requirements
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
CMD /bin/bash